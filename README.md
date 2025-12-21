# surface_reconstruction


## Introduction

由作者于自身对于三维重建比较感兴趣，于是大三上从11月末开始着手看三维重建的论文以及相关源码，这个仓库是作者在做了以上这些事后，结合自己对于三维重建的理解和别人的思想，重新写的一个三维重建项目.

## Probelms And Thinking

1. 作者使用的是orbbec femto bolt,这款相机是根据tof原理测量深度的，由于是一款大众级消费深度相机，所以精度有限，如果不进行标定和深度矫正，那么使用fpfh+ransca的粗匹配基本都会失败.
