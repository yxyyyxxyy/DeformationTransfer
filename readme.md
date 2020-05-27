# Deformation Transfer

一个简单的实现，使用了 intel math kernel library 和 openmp 加速

```
./dt <s0_obj_name> <s1_obj_name> <t0_obj_name> <t1_obj_name>
```

按照 S0 到 S1 的变形，将 t0 变为 t1 并输出。输入的三个 `.obj` 模型的各个顶点需要是已对齐的。