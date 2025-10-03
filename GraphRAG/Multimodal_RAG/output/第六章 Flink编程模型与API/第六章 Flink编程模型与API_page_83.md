```groovy
}
}

override def onTimer(timestamp: Long, ctx: KeyedProcessFunction[I, StationLog, String]#OnTimerCont
    //定时器触发时,说明该被叫手机号连续5s呼叫失败,输出告警信息
    out.collect("触发时间:" + timestamp + " 被叫手机号:" + ctx.getCurrentKey + " 连续5秒呼叫失败!")
    //清空时间状态
    timeState.clear()
}
}
).print()
env.execute()
```

# 6.10 Flink 异步IO机制

## 6.10.1 异步IO机制原理

Flink的异步I/O是一个非常受欢迎的特性,由阿里巴巴贡献给社区,并在1.2版本中引入,它的主要目的是解决与外部系统交互时网络延迟成为系统瓶颈的问题,外部系统往往是外部数据库。

在Flink流计算系统中,与外部数据库进行交互是常见的需求,通常情况下,我们会发送一个查询请求到数据库并等待结果返回,这期间无法发送其他请求,这种同步访问方式会导致阻塞,阻碍了吞吐量和延迟,为了解决这个问题,引入了异步模式,能够并发地处理多个到外部数据库的请求,下图为官方提供的异步IO原理图。