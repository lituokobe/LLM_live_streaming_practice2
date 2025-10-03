* **EventTime**\*\* 支持：\*\*Storm早期和SparkStreaming实时数据处理不支持事件时间，Storm后期实时数据处理支持事件时间，同样Spark后期推出的StructuredStreaming处理流数据也是支持事件时间，Flink诞生开始处理实时数据就支持事件时间。
* **保证次数**：在数据处理方面，Storm可以实现至少处理一次，但不能保证仅处理一次，这样就会导致数据重复处理问题，所以针对计数类的需求，可能会产生一些误差，SparkStreaming、StructuredStreaming和Flink支持Exactly-once数据处理语义。
* **容错机制**：Storm可以通过ACK机制实现数据的容错机制，而SparkStreaming、StructuredStreaming和Flink可以通过CheckPoint机制实现容错机制。
* **状态管理**：Storm中没有实现状态管理，SparkStreaming实现了基于DStream的状态管理，StructuredStreaming支持基于Dataset/DataFrame的状态管理，而Flink实现了基于操作的状态管理。
* **延时**：表示数据处理的延时情况，Storm和Flink接收到一条数据就处理一条数据，其数据处理的延时性是很低的；SparkStreaming和StructuredStreaming都支持微批处理，数据处理的延时性相对会偏高，虽然StructuredStreaming支持Continuous连续处理，但是目前处于实验阶段，数据处理延迟性相对Flink偏高，Flink实时数据处理延迟最低。
* **吞吐量**：Storm的吞吐量其实也不低，只是相对于其他几个框架而言较低；SparkStreaming、StructuredStreaming和Flink的吞吐量是比较高的。