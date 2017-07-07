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

object KMeans {
  val TILE_SIZE = 256

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
    def splitTif(filePath: String): RDD[(ProjectedExtent, Tile)] = sc.hadoopGeoTiffRDD(filePath).split(TILE_SIZE, TILE_SIZE)

    val tifExtent = sc.hadoopGeoTiffRDD("file:///Users/jnachbar/Downloads/LC80160342016111LGN00_B2.TIF").first()._1.extent

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
    val df = tupleRDD.toDF("pe", "col", "row", "features").repartition(200)
    //val schemaString = "Col Row"
      //"Tile1 Tile2 Tile3"
    //val labels = schemaString.split(" ").map(name => StructField(name, types.IntegerType, nullable = true))
    //val schema = StructType(labels)
    //make it so that I go through every tuple, not just every sequence
    //val rowSeq = tupleRDD.first().map(sequence => (sequence._1, sequence._2))
    //val rowRDD = sc.parallelize(rowSeq.map(value => Row(value._1, value._2)))
      //sequence.apply(2), sequence.apply(3),
      //sequence.apply(4), sequence.apply(5)))
    //val rawData = sparkSession.createDataFrame(rowRDD, schema)
    //rawData.show(5)
    //create function to explode tiles into pixel/key pairs
    val kmeans = new KMeans().setFeaturesCol("features").setK(6)
    val model = kmeans.fit(df)
    val preds = model.transform(df)
    preds.groupBy("prediction").count().show()
    //start with an RDD with partitions based on one of the 900 possible extents

    //take a sequence and make it into an array
    def iterArray(seq: Seq[(Extent, Int, Int, Double)]): Array[Double] = {
      val valArr = Array.ofDim[Double](TILE_SIZE * TILE_SIZE)
      for(i: Int <- 0 until (TILE_SIZE * TILE_SIZE))
        valArr(i) = seq.apply(i)._4
      valArr
    }

    //grabs what we need out of the dataFrame
    val predsRDD = preds.select("pe", "col", "row", "prediction")
      .as[(Extent, Int, Int, Double)].rdd
      .groupBy(_._1)

    //creates an RDD with an array in it
    val arrRDD = predsRDD
      .map(value => (value._1, iterArray(value._2.toSeq)))

    /** This takes care of packing a tile. */
    val makeTile = (pair: (Extent, Array[Double])) => (pair._1, ArrayTile(pair._2, TILE_SIZE, TILE_SIZE))

    //The number of tiles in each direction. In this case, 30
    val NUMBER_OF_TILES = 30

    val tl = TileLayout(NUMBER_OF_TILES, NUMBER_OF_TILES, TILE_SIZE, TILE_SIZE)
    val layout = LayoutDefinition(tifExtent, tl)

    //Assigns the tiles 2D indexes based on position relative to the larger extent
    def indexTile(pair: (Extent, Tile)): ((Int, Int), Tile) = {
      val gridBounds = layout.mapTransform(pair._1)
      println(gridBounds)
      ((gridBounds.colMin, gridBounds.colMax), pair._2)
    }

    //performs transformations before the stitcher
    val arrTile = arrRDD
      .map(makeTile)
      .map(indexTile)
      .toLocalIterator
      .toSeq

    //Stitches the tiles together
    val stitchTile = TileLayoutStitcher.stitch[Tile](arrTile)
    println(stitchTile._1.cols)
    println(stitchTile._1.rows)

    //val stitchTile = layout.mapTransform({???}: Extent)

    //arrTile.head._2.renderPng(ColorRamps.HeatmapBlueToYellowToRedSpectrum).write(s"//Users/jnachbar/Documents/Pictures/tileOne.png")
  }
}
