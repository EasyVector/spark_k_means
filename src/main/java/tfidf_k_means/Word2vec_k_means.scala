package tfidf_k_means

import org.apache.spark.Partitioner
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.SparkSession

/**
 * word2vec表征文本特征，k-means实现文本聚类
 * written with scala
 * Created by Su yu hui on 21-05-01
 */

object Word2vec_k_means {
  def main(args: Array[String]) {
    val path = "src/main/resources/20news-bydate-train/*"

    val sqlContext = SparkSession.builder().appName("word2vec").master("local[*]").getOrCreate()

    import sqlContext.implicits._ // import toDF and convert rdd implicitly

    val rdd = sqlContext.sparkContext.wholeTextFiles(path)  // read all the files whose file path matches the path above

    println(rdd.count)  // get the count of document

    val text = rdd.map { case (file, text) => text }  // Tokenize the text data

    val regex = """[^0-9]*""".r

    val stopwords = Set(
      "the","a","an","of","or","in","for","by","on","but", "is", "not", "with", "as", "was", "if",
      "they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to"
    )

    // split text on any non-word tokens
    val nonWordSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase))

    // filter out numbers
    val filterNumbers = nonWordSplit.filter(token => regex.pattern.matcher(token).matches)

    // get distinct words
    val tokenCounts = filterNumbers.map(t => (t, 1)).reduceByKey(_ + _)

    // get rare words
    val rareTokens = tokenCounts.filter{ case (k, v) => v < 2 }.map{ case (k, v) => k }.collect.toSet

    // create a function to tokenize each document
    def tokenize(line: String): Seq[String] = {
      line.split("""\W+""")  // remove all non-word tokens
        .map(_.toLowerCase)  // convert to number
        .filter(token => regex.pattern.matcher(token).matches) // remove all the numbers
        .filterNot(token => stopwords.contains(token))  // remove all stop words
        .filterNot(token => rareTokens.contains(token))  // remove all rare tokens
        .filter(token => token.length >= 2)  // all token should be longer than or equal to 2
        .toSeq  // get a sequence according to a row
    }

    println(text.flatMap(doc => tokenize(doc)).distinct.count) // the dictionary size

    val tokens = rdd.map { case (file, text) => (file.split("/").takeRight(2).head, text) }.map{case(file, doc) => (file,tokenize(doc))}

    val docs = tokens.toDF("type","sentence_words")  // make a data frame

    // get word2vec feature
    val word2vec = new Word2Vec().setInputCol("sentence_words").setOutputCol("rawFeatures").setVectorSize(100).setMinCount(2)
    val model_word2vec = word2vec.fit(docs)
    val dfWord2Vec = model_word2vec.transform(docs)

    // k-means model
    // train a k-means model
    val kmeans = new KMeans().setK(6).setSeed(1L).setFeaturesCol("rawFeatures").setPredictionCol("predicted_cluster")
    val model_kmeans = kmeans.fit(dfWord2Vec)
    val dfClustered = model_kmeans.transform(dfWord2Vec)

    // dfClustered的schema：|type|sentence_words|rawFeatures|predicted_cluster|
    // 获取每一行文档的文档类型以及此文档被分入的簇的序号
    val outputRdd = dfClustered.rdd.map(row =>(row.getString(0),row.getInt(3)))//type|predicted_cluster

    // 获得文档类型，并为某类文档绑定一个序号，例如comp.graphics -> 2, alt.atheism -> 1，依次划分partition
    val fileNameFirstCharMap = outputRdd.map(_._1.toString).distinct().zipWithIndex().collect().toMap

    // 根据文档类型以及序号关系，将rdd分区
    val partitionData = outputRdd.partitionBy(FileNamePartitioner(fileNameFirstCharMap))

    // 首先，根据文档类型和分入的簇，建立一个元组((document_type, cluster_index),1)
    // 根据不同的(document_type, cluster_index)，用aMap进行计数
    val combined = partitionData.map(x =>( (x._1, x._2),1) ).mapPartitions{f => var aMap = Map[(String,Int),Int]();
      for(t <- f){
        if (aMap.contains(t._1)){
          aMap = aMap.updated(t._1,aMap.getOrElse(t._1,0)+1)
        }else{
          aMap = aMap + t
        }
      }
      val aList = aMap.toList
      val total= aList.map(_._2).sum // 获取这个文档类型内文档的总数量
      val group = aMap.maxBy(_._2)._1._2  // 根据文档被划分进的簇进行计数，选取拥有最多的文档的那个簇序号
      val total_right = aList.map(_._2).max  //  查看拥有最多的文档的那个簇序号下的文档数量
      List((aList.head._1._1,total,total_right,group)).toIterator
    }
    var result = combined.collect()
    result = result.sortBy(_._4)
    for(re <- result ){
      println("文档类型："+re._1+"；文档总数："+ re._2+"；大多数此类文档被分入的簇为："+ re._4+"；分入此簇的文档数量："+re._3+"；这个簇内的文档数与此类文档的数量的比例是："+(re._3*100.0/re._2)+"%")
    }
    //output:
    //文档类型：soc.religion.christian；文档总数：599；大多数此类文档被分入的簇为：0；分入此簇的文档数量：358；这个簇内的文档数与此类文档的数量的比例是：59.76627712854758%
    //文档类型：talk.religion.misc；文档总数：377；大多数此类文档被分入的簇为：0；分入此簇的文档数量：136；这个簇内的文档数与此类文档的数量的比例是：36.07427055702918%
    //文档类型：sci.space；文档总数：593；大多数此类文档被分入的簇为：1；分入此簇的文档数量：158；这个簇内的文档数与此类文档的数量的比例是：26.644182124789207%
    //文档类型：alt.atheism；文档总数：480；大多数此类文档被分入的簇为：4；分入此簇的文档数量：185；这个簇内的文档数与此类文档的数量的比例是：38.541666666666664%
    //文档类型：rec.motorcycles；文档总数：598；大多数此类文档被分入的簇为：5；分入此簇的文档数量：160；这个簇内的文档数与此类文档的数量的比例是：26.755852842809364%
    //文档类型：talk.politics.guns；文档总数：546；大多数此类文档被分入的簇为：5；分入此簇的文档数量：209；这个簇内的文档数与此类文档的数量的比例是：38.27838827838828%
    //文档类型：sci.med；文档总数：594；大多数此类文档被分入的簇为：5；分入此簇的文档数量：145；这个簇内的文档数与此类文档的数量的比例是：24.410774410774412%
    //文档类型：talk.politics.misc；文档总数：465；大多数此类文档被分入的簇为：5；分入此簇的文档数量：154；这个簇内的文档数与此类文档的数量的比例是：33.11827956989247%
    //文档类型：misc.forsale；文档总数：585；大多数此类文档被分入的簇为：6；分入此簇的文档数量：157；这个簇内的文档数与此类文档的数量的比例是：26.837606837606838%
    //文档类型：rec.sport.baseball；文档总数：597；大多数此类文档被分入的簇为：11；分入此簇的文档数量：238；这个簇内的文档数与此类文档的数量的比例是：39.86599664991625%
    //文档类型：rec.sport.hockey；文档总数：600；大多数此类文档被分入的簇为：11；分入此簇的文档数量：287；这个簇内的文档数与此类文档的数量的比例是：47.833333333333336%
    //文档类型：comp.graphics；文档总数：584；大多数此类文档被分入的簇为：12；分入此簇的文档数量：132；这个簇内的文档数与此类文档的数量的比例是：22.602739726027398%
    //文档类型：comp.windows.x；文档总数：593；大多数此类文档被分入的簇为：12；分入此簇的文档数量：280；这个簇内的文档数与此类文档的数量的比例是：47.21753794266442%
    //文档类型：comp.os.ms-windows.misc；文档总数：591；大多数此类文档被分入的簇为：12；分入此簇的文档数量：179；这个簇内的文档数与此类文档的数量的比例是：30.287648054145517%
    //文档类型：talk.politics.mideast；文档总数：564；大多数此类文档被分入的簇为：13；分入此簇的文档数量：223；这个簇内的文档数与此类文档的数量的比例是：39.53900709219858%
    //文档类型：sci.electronics；文档总数：591；大多数此类文档被分入的簇为：14；分入此簇的文档数量：162；这个簇内的文档数与此类文档的数量的比例是：27.411167512690355%
    //文档类型：rec.autos；文档总数：594；大多数此类文档被分入的簇为：14；分入此簇的文档数量：194；这个簇内的文档数与此类文档的数量的比例是：32.65993265993266%
    //文档类型：comp.sys.ibm.pc.hardware；文档总数：590；大多数此类文档被分入的簇为：15；分入此簇的文档数量：248；这个簇内的文档数与此类文档的数量的比例是：42.03389830508475%
    //文档类型：comp.sys.mac.hardware；文档总数：578；大多数此类文档被分入的簇为：15；分入此簇的文档数量：193；这个簇内的文档数与此类文档的数量的比例是：33.391003460207614%
    //文档类型：sci.crypt；文档总数：595；大多数此类文档被分入的簇为：19；分入此簇的文档数量：228；这个簇内的文档数与此类文档的数量的比例是：38.319327731092436%
  }
}


/**
 * 根据文件名的第一个字符来分区
 * @param fileNameFirstCharMap
 */
case class FileNamePartitioner(fileNameFirstCharMap:Map[String,Long]) extends Partitioner{
  override def getPartition(key: Any): Int = key match {
    case _ => fileNameFirstCharMap.getOrElse(key.toString,0L).toInt
  }
  override def numPartitions: Int = fileNameFirstCharMap.size
}