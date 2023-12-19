导出conda环境 `conda env export --no-builds > environment.yaml`

torchserve 需要java11

https://github.com/pytorch/serve/issues/473

1.创建模型存档mar文件

把所有的代码打包到model.bin文件中,然后执行如下代码,

提前安装torchserve

```
torch-model-archiver --model-name image2floorplan \
                    --version 5.0 \
                    --serialized-file /path/to/your/model.bin \
                    --handler /path/to/your/handler.py \
                    --export-path /path/to/your/output
```

2.启动torchserve

将上一步生成的mar文件(这就是个压缩文件)

```
torchserve --start --model-store=./image2floorplan --models image2floorplan.mar

```

