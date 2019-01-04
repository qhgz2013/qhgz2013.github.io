---
layout: post
title:  "在Github上使用verified sign进行提交"
date:   2019-01-04 13:27:41 +0800
categories: github
tags: github
---

* content
{:toc}

在Github上使用verified sign进行提交




## 0. 前言
确保在Windows上已经安装了Git Bash

## 1. 检查GPG key
```bash
gpg --list-secret-keys --keyid-format LONG
```
如果执行该命令有输出，则跳过第二步

## 2. 生成GPG key
1. 先检查gpg的版本
```bash
gpg --version
```
2. 如果gpg版本大于2.1.17，则直接执行
```bash
gpg --full-generate-key
```
否则执行下面的命令并跳到第7步
```bash
gpg --default-new-key-algo rsa4096 --gen-key
```
3. 直接按下`Enter`，选择`RSA and RSA (default)`
4. RSA keysize输入最大的keysize `4096`
5. 直接按下`Enter`，选择`0 = key does not expire`
6. 检查信息是否正确
7. 输入Real name、Email address和Comment信息，填写GitHub上验证过的邮箱地址
8. 输入passphrase，记好这个密码
9. 再次执行`gpg --list-secret-keys --keyid-format LONG`查看公钥和私钥

## 3. 添加GPG key
1. 在输出中复制GPG key ID，如下面输出的`3AA5C34371567BD2`
```bash
$ gpg --list-secret-keys --keyid-format LONG
/Users/hubot/.gnupg/secring.gpg
------------------------------------
sec   4096R/3AA5C34371567BD2 2016-03-10 [expires: 2017-03-10]
uid                          Hubot 
ssb   4096R/42B317FD4BA89E7A 2016-03-10
```
2. 运行下面代码，把`3AA5C34371567BD2`替换成自己的key
```bash
$ gpg --armor --export 3AA5C34371567BD2
# Prints the GPG key ID, in ASCII armor format
```
3. 复制GPG key，从`-----BEGIN PGP PUBLIC KEY BLOCK-----`开始复制到`-----END PGP PUBLIC KEY BLOCK-----`
4. 将GPG key添加到GitHub上

## 4. 添加GPG key到GitHub
1. 在个人的GitHub头像上选择**Settings**
2. 点击**SSH and GPG keys**
3. 点击**New GPG key**
4. 在`Key`一栏中，粘贴生成的GPG key
5. 点击**Add GPG key**
6. 输入github的密码进行确认
7. 在本地提交中输入`git config --global user.signingkey 3AA5C34371567BD2`，配置全局开启GPG验证
8. 在每次提交时输入passphrase即可
