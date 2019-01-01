<div align="center">
    <img src="doc/LogoMakr_0v1ngq.png" width="250px">
</div>

<!-- GFM-TOC -->
* [Preview](#preview)
* [Introduction](#introduction)
* [Usage](#usage)
<!-- GFM-TOC -->

# Preview

![](doc/1.gif)


# Introduction

- Replace `[TOC]` tag into generated catalogue.
- Use CodeCogs to display Mathjax formula.
- Center images.
- Escape some spacial characters.

# Usage

You can just run App.java, and then specify the path of Markdown document. After that, a new file suffixed with `.gfm` will be generated.

```java
javac -encoding UTF-8 App.java
java App
```

You can also use below API in your program.

```java
GFM.convert(srcFilePath, detFilePath);
```
