import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SparkSession, types}
import geotrellis.spark._
import geotrellis.spark.io.hadoop._
import geotrellis.raster._
import geotrellis.raster.histogram._
import geotrellis.raster.render.ColorRamps
import geotrellis.vector.{Extent, ProjectedExtent}
import org.apache.spark.rdd.RDD
import Utils._
import geotrellis.spark.stitch.TileLayoutStitcher
import geotrellis.spark.tiling.LayoutDefinition
import org.apache.avro.generic.GenericData.StringType
import org.apache.spark
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.types.DateType._
import org.apache.spark.sql.types.{BinaryType, StructField, StructType}

/**
  * Created by jnachbar on 6/20/17.
  */

//process:
//1. Collect data into one RDD
//2. Expand the RDD structure to make it easier to map
//3. Get a sequence of sequences with (index, row, column, data1, data2, data3)
//4. Turn the first three values into an identifier and the last three into a feature vector
//5. Turn the whole structure into a dataframes
//6. Run K-Means
//7. Reconstruct the image using your predefined index

//to do:
//try stitching together 15 * 15 empty tiles

object KMeans {

  //Making a tuple with all the info that I need
  def tilesToKeyPx(pe: ProjectedExtent, t1: (Tile), t2: (Tile), t3: (Tile)) = {
    for {
      r <- 0 until t1.rows
      c <- 0 until t1.cols
      v = t1.get(c, r)
      if v > 0
    } yield(pe.extent, c, r, Vectors.dense(t1.get(c, r), t2.get(c, r), t3.get(c, r)))
          //r, t1.get(c, r), t2.get(c, r), t3.get(c, r))
  }

  //Function for expanding the inconveniently zipped tuple of tiles
  def expandTuple(t: ((Tile, Tile), Tile)) = (t._1._1,  t._1._2, t._2)

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("IrisSpark")
    val sparkSession = SparkSession.builder
      .appName("KMeans")
      .master("local[*]")
      .getOrCreate()

    import sparkSession.implicits._

    val sc = sparkSession.sparkContext
    // imports a TIF and then splits it

    val tif = sc.hadoopGeoTiffRDD("file:///Users/jnachbar/Downloads/LC80160342016111LGN00_B2.TIF").first()


    //try three for columns and 23 for rows
    val COL_SIZE = tif._2.cols/3
    val ROW_SIZE = tif._2.rows/23


    def splitTif(filePath: String): RDD[(ProjectedExtent, Tile)] = sc.hadoopGeoTiffRDD(filePath).split(COL_SIZE, ROW_SIZE)

    val blue: RDD[(ProjectedExtent, Tile)] = splitTif("file:///Users/jnachbar/Downloads/LC80160342016111LGN00_B2.TIF")
    val green: RDD[(ProjectedExtent, Tile)] = splitTif("file:///Users/jnachbar/Downloads/LC80160342016111LGN00_B3.TIF")
    val red: RDD[(ProjectedExtent, Tile)] = splitTif("file:///Users/jnachbar/Downloads/LC80160342016111LGN00_B4.TIF")

    //create a multi-band tuple
    val expandRDD = (blue.zip(green).zip(red))
        .map { case((t1,t2),t3) => (t1, t2, t3) }

    val tupleRDD = expandRDD.flatMap { case ((pe1: ProjectedExtent, value1: Tile),
    (pe2: ProjectedExtent, value2: Tile),
    (pe3: ProjectedExtent, value3: Tile)) => tilesToKeyPx(pe1, value1, value2, value3)
    }

    val df = tupleRDD.toDF("pe", "col", "row", "features").repartition(200).toDF()

    //create function to explode tiles into pixel/key pairs
    val kmeans = new KMeans().setFeaturesCol("features").setK(6)
    val model = kmeans.fit(df)
    val preds = model.transform(df)
    //preds.groupBy("prediction").count().show()
    //start with an RDD with partitions based on one of the 900 possible extents

    //take a sequence and make it into an array

    def iterArray(seq: Seq[(Extent, Int, Int, Double)]): Array[Double] = {
      val valArr = Array.ofDim[Double](seq.length)
      for(i: Int <- 0 until seq.length)
        //this line must be changed to avoid indexOutOfBounds
        valArr(i) = seq.apply(i)._4
      valArr
    }

    //grabs what we need out of the dataFrame
    //somehow sort this to be in order
    //rows + columns - max
    val distance = (pair: (Extent, Array[Double])) => (pair._1, ArrayTile(pair._2, COL_SIZE, ROW_SIZE))

    val predsRDD = preds.select("pe", "col", "row", "prediction")
      .as[(Extent, Int, Int, Double)].rdd
      .groupBy(_._1)

    println(predsRDD.count())

    //creates an RDD with an array in it
    val arrRDD = predsRDD
      .map(value => (value._1, iterArray(value._2.toSeq)))

    /** This takes care of packing a tile. */
    val makeTile = (pair: (Extent, Array[Double])) => (pair._1, ArrayTile(pair._2, COL_SIZE, ROW_SIZE))

    //this is a gridExtent stating that the
    val ge = GridExtent(tif._1.extent, 3.toDouble, 23.toDouble)
    val layout = LayoutDefinition(ge, COL_SIZE, ROW_SIZE)

    //Assigns the tiles 2D indexes based on position relative to the larger extent
    def indexTile(pair: (Extent, Tile)): ((Int, Int), Tile) = {
      val gridBounds = layout.mapTransform(pair._1)
      //should it be rowMin?
      ((gridBounds.colMin, gridBounds.colMax), pair._2)
    }

    //performs transformations before the stitcher
    val arrTile = arrRDD
      .map(makeTile)
      .map(indexTile)
      .sortBy(value => value._1)
      .toLocalIterator
      .toSeq

    //try stitching together empty tiles

    for(i <- arrTile.indices)
      println(arrTile.apply(i)._1)
      println(arrTile.length)
    //Stitches the tiles together
    //this is where everything goes wrong
    val stitchTile = TileLayoutStitcher.stitch[Tile](arrTile)
    println(stitchTile._1.cols)
    println(stitchTile._1.rows)

    //val stitchTile = layout.mapTransform({???}: Extent)

    //arrTile.head._2.renderPng(ColorRamps.HeatmapBlueToYellowToRedSpectrum).write(s"//Users/jnachbar/Documents/Pictures/tileOne.png")

  }
}
