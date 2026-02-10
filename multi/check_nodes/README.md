## 节点筛查(此次检查是基于GPT-MOE 567B模型单机参数)

```shell
1、到multi/check_nodes目录下，将要筛查的节点写入clushnode文件
2、bash clush.sh，检查环境基本情况，如显存、内存等是否已释放
3、打开check_nodes.sh，将基本环境变量补齐或做相应修改
4、bash run_check.sh 1/4，进行单机或者四机的节点筛查 # 当前只支持单机和四机筛查
```

