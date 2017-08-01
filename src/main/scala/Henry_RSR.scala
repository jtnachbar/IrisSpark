/**
  * Created by Henry on 7/21/17.
  */
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import scala.io.Source

object RSR_Diff {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RSR").setMaster("local[2]")

    val sc = new SparkContext(conf)

    val sparkSession = SparkSession.builder
      .config(conf = conf)
      .appName("RSR")
      .getOrCreate()

    import sparkSession.implicits._

    val file = "/Users/jnachbar/Downloads/Jamie_rsr.1.oobd.det"

    val rows = Source.fromFile(file)
      .getLines()
      .filter(!_.startsWith("#"))
      .map(line => {
        val parts = line.split("[ ]+").filter(_.nonEmpty)
        (parts(0).toInt, parts(1).toInt, parts(2).toDouble, parts(3).toDouble)
      })
      .toList

    val data = rows.toDF("band", "channel", "wavelength", "rsr").repartition(4)
    data.printSchema()
    data.show()

    val cols = data.columns.map(col)
    data.agg(max(cols.head), cols.tail.map(max): _*).show()

    val df = data.withColumn("lagged_column", (lag($"wavelength", 1))
      .over(Window.partitionBy("channel").orderBy($"wavelength")))

    df.show(10)
    val delta = df.withColumn("delta", df("wavelength") - df("lagged_column"))
    delta.select(avg("delta")).show //Calculates the average distance between the dataset's wavelengths
//
    val lambda_c = delta.withColumn("product", delta("wavelength") * delta("rsr"))
//
    val wave_calc = lambda_c.agg(sum("product").alias("sum_product"),
      sum("rsr").alias("sum_rsr"))
    wave_calc.withColumn("expected_wavelength", wave_calc("sum_product")/wave_calc("sum_rsr")).show

  }
}
