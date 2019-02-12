---
layout: post
title:  "Pixiv的日榜爬虫"
date:   2019-01-21 22:33:24 +0800
categories: crawler
tags: pixiv
---

* content
{:toc}




## Pixiv日榜爬虫的原理与实现

### 原理

总结起来就两个字：抓包

首先，打开fiddler，如常访问[p站日榜]，如下图：  
![1](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_1.png)

然后就可以套上BeautifulSoup直接解析html了。不过我后来发现了抓到一个这样的包，是在加载日榜51-100项的时候发出的请求。  
![3](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_3.png)

标准的json格式，连html解析都不用了。打开contents一看，内容一目了然，ID、页数、标题、图片URL、作者、Tag等，一堆有用信息都已经显示出来了，所以直接拿出来用，比解析html更高效且信息量更全。  
![4](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_4.png)

请求参数也很简单，`mode=daily`不需要改，`p=2`看上去是分页，1=1~50的内容，2=51~100的内容，以此类推，`format=json`也是固定好的，只有一个`tt=382e6eae9796bc30389512c4674e05de`是需要考虑怎样得到的。  
在抓包历史中搜索`382e6eae9796bc30389512c4674e05de`，可以看到之前的日榜网页被标记了，说明这个字符串可以直接通过网页获得  
![5](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_5.png)

在网页上查找一下，不难发现有一行是`pixiv.context.token = "382e6eae9796bc30389512c4674e05de";`：  
![2](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_2.png)

这就好办了嘛，套一个正则就可以拿出来了，正则大概长这样：`pixiv\.context\.token\s*=\s*"(\w+)";`，匹配完后直接用`group(1)`就能得到了。  

验证一下参数`p`，的确如所想的那样，而且还发现了更改日期只要改`date`就好了  
![7](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_7.png)

然后就看图片了，用上图的url看看抓包抓到的是什么。  
![8](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_8.png)

嗯，没错，而且要注意了，左边的`Referer`可是不能漏的，不然就这样：  
![9](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_9.png)

恶意满满的403~

所以从json数据中获得的一个图片缩略图的url长这样，如上图所示：  
`https://i.pximg.net/c/240x480/img-master/img/2019/01/17/23/28/48/72712034_p0_master1200.jpg`

感觉有点小，残念，点进去看会更清楚一点（前面那张图被换掉了，因为戳进去显示正在浏览敏感图片emm）  
![10](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_10.png)

这时候的url变成这样  
`https://i.pximg.net/c/600x600/img-master/img/2019/01/17/23/28/48/72712034_p0_master1200.jpg`

找到规律了吧，把`/c/?x?/img-master/...`中的`?`改成更大的数值的话，就能获取更清晰的图片了~

当然，注册个账号登陆进去的话，图片就变得更大了：  
![11](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_11.png)

这时候url就是  
`https://i.pximg.net/img-master/img/2019/01/17/23/28/48/72712034_p0_master1200.jpg`

点击查看原图，url就变成下面这样，过程就不贴图了  
`https://i.pximg.net/img-original/img/2019/01/17/23/28/48/72712034_p0.png`

综上所述，图片的格式差不多摸透了，剩下的`/c/`后面能接多少就靠自己发现吧。  
**缩略图** （里面的`?`自己摸索吧，上面已经有`240x480`和`600x600`的了）：  
`https://i.pximg.net/c/?x?/img-master/img/2019/01/17/23/28/48/72712034_p0_master1200.jpg`  
**大图** （大于1000px）：  
`https://i.pximg.net/img-master/img/2019/01/17/23/28/48/72712034_p0_master1200.jpg`  
**原图** ：  
`https://i.pximg.net/img-original/img/2019/01/17/23/28/48/72712034_p0.png`

替换json中的图片也就是一条正则的事，下面贴出代码（这段代码也包含在最后的代码中）。  
两个参数，一个是url，就是上面从json得到的`https://i.pximg.net/img-master/img/2019/01/17/23/28/48/72712034_p0_master1200.jpg`  
另外一个是page，指定要爬第几张图片（针对多图投稿），第一张图就是`0`

```python
import re
from warning import warn
def replace_url(url, page):
        url_pattern = re.compile(r'(?P<schemas>https?)://(?P<host>([^./]+\.)+[^./]+)(/c/\d+x\d+)?'
                                 r'(?P<path_prefix>/img-master/img(/\d+){6}/\d+_p)\d+'
                                 r'(?P<path_postfix>_(master|square)\d+\.(jpg|png)).*')
        match = re.match(url_pattern, url)
        if match:
            schemas = match.group('schemas')
            host = match.group('host')
            path_prefix = match.group('path_prefix')
            path_postfix = match.group('path_postfix')
            return '%s://%s%s%d%s' % (schemas, host, path_prefix, page, path_postfix)
        url_pattern = re.compile(r'(?P<schemas>https?)://(?P<host>([^./]+\.)+[^./]+)(/c/\d+x\d+)?'
                                 r'(?P<path_prefix>/img-master/img(/\d+){6}/\d+)'
                                 r'(?P<path_postfix>_(master|square)\d+\.(jpg|png)).*')
        match = re.match(url_pattern, url)
        if match:
            schemas = match.group('schemas')
            host = match.group('host')
            path_prefix = match.group('path_prefix')
            path_postfix = match.group('path_postfix')
            if page != 0:
                warn('A non-pageable image url detected, your page should be 0 constantly, but got %d' % page)
            return '%s://%s%s%s' % (schemas, host, path_prefix, path_postfix)

        raise ValueError('The url "%s" could not match any replacement rules' % url)
```

爬的时候才发现：有些图中间的`_p0`是没有的，也就是直接剩下了`72712034_master1200.jpg`，这里也要注意一下，不然一不留神就出错了。

当然比较坑的是，原图是有png格式的，这东西在未登陆的时候比较难知道，所以要花费更多的时间在试图寻找jpg或png上。

### 程序

一大串python代码，兼容python 3.5和3.6  
~~写多线程爬虫就别纠结代码的整体美观了~~  
爬的是全部的日榜的大图（非原图），Top 100，默认使用privoxy和ss代理（代理需要自己配置，如不需要则改为`proxy = None`），默认使用5线程下载  
需要自己改`save_path`指定保存的位置，这段代码会生成1000个文件夹，按id尾数分开存储，如`save_path\777`文件夹保存的都是尾数是`777`的图片

代码我是部署在树莓派上的，为了提升速度，做了挺多的内存缓存的，所以吃掉500M的内存，每天更新大概只要花20分钟左右（10分钟爬取，10分钟更新数据库）  
到目前为止，这个数据集有246G大小，有693k个文件
![13](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_13.png)

数据库表说明：  
`user`：用户表，存有用户id、名称及头像url  
`illust_series`：投稿的系列作品，这个是作者在投稿时指定的，存有系列id、创建用户id、标题、简介、属于本系列的投稿数量、创建时间和系列的url  
`illust`：插画，存有标题、投稿时间、图片url、illust_type（未知）、book_style（未知）、页数、内容类型（如原创、暴力、X暗示等）、系列ID（不存在时为null）、id、宽高（多页投稿时默认指第一页）、用户id、评分数、浏览数、上传时间、属性（内容类型对应的string）  
`tag`：插画标签，存有标签的id（自增字段）和标签名  
`illust_tags`：插画-标签的关系表，一个插画对应多个标签，一个标签对应多个插画，存有标签id和插画id  
`illust_rank`：插画的排行信息，存有插画id、时间、当前排名和昨日排名

```python
import requests
from datetime import datetime, timedelta
import threading
import json
import re
from hashlib import md5
import os
from warnings import warn
import sqlite3
import numpy as np
import pickle
from tqdm import tqdm


ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) ' \
     'Chrome/53.0.2785.143 Safari/537.36'
# path to save images from pixiv
save_path = '/share/disk/ML-TRAINING-SET/PixivRanking'
# save_path = 'd:/ML-TRAINING-SET/PixivRanking'
# path to save ranking data cache
cache_path = os.path.join(save_path, '.cache')
# path to generate sqlite database
db_path = os.path.join(save_path, 'database.db')
# proxy, for those who could not access pixiv directly
proxy = {'https': 'https://localhost:8118'}


def calc_md5(str_data):
    hash_obj = md5()
    hash_obj.update(str_data.encode('utf8'))
    return hash_obj.hexdigest()


def create_dir(path):
    parent = os.path.abspath(path)
    dir_to_create = []
    while not os.path.exists(parent):
        dir_to_create.append(parent)
        parent = os.path.abspath(os.path.join(parent, '..'))
    dir_to_create = dir_to_create[::-1]
    for dir_path in dir_to_create:
        os.mkdir(dir_path)
        print('Directory %s created' % dir_path)


class FileCacher:
    def __init__(self):
        self._cache_files = dict()
        self._lock = threading.RLock()

    def add_cache_dir(self, directory, create_dir_if_not_exist=True):
        with self._lock:
            path = os.path.abspath(directory)
            if os.path.exists(path):
                files = set(os.listdir(path))
            else:
                files = set()
                if create_dir_if_not_exist:
                    create_dir(path)
            self._cache_files[path] = files

    def append_file(self, file_path):
        with self._lock:
            dir_path = os.path.abspath(os.path.join(file_path, '..'))
            files = self._cache_files.get(dir_path, None)
            if files is None:
                warn('%s is not in the cached directory, calling add_cache_dir implicitly' % dir_path)
                self.add_cache_dir(dir_path, True)
                files = self._cache_files.get(dir_path, None)
                assert files is not None
            file_name = os.path.basename(file_path)
            self._cache_files[dir_path].add(file_name)
    
    def remove_file(self, file_path):
        with self._lock:
            dir_path = os.path.abspath(os.path.join(file_path, '..'))
            files = self._cache_files.get(dir_path, None)
            if files is None:
                warn('%s is not in the cached directory, calling add_cache_dir implicitly' % dir_path)
                self.add_cache_dir(dir_path, True)
                files = self._cache_files.get(dir_path, None)
                assert files is not None
            file_name = os.path.basename(file_path)
            self._cache_files[dir_path].remove(file_name)

    def exist_file(self, file_path):
        with self._lock:
            dir_path = os.path.abspath(os.path.join(file_path, '..'))
            files = self._cache_files.get(dir_path, None)
            if files is None:
                warn('%s is not in the cached directory, calling add_cache_dir implicitly' % dir_path)
                self.add_cache_dir(dir_path, True)
                files = self._cache_files.get(dir_path, None)
            file_name = os.path.basename(file_path)
            return file_name in files

    def exist_dir_in_cache(self, dir_path):
        with self._lock:
            return self._cache_files.get(os.path.abspath(dir_path), None) is not None

    def validate_dir(self, dir_path):
        with self._lock:
            dir_path = os.path.abspath(dir_path)
            files = self._cache_files.get(dir_path, None)
            if files is not None:
                files = set(files)
                actual_files = set(os.listdir(dir_path))
                same_file_count = len(files.intersection(actual_files))
                is_same = len(files) == same_file_count and len(actual_files) == same_file_count
                if not is_same:
                    warn('cache inconsistency detected in directory %s, cleared all cache' % dir_path)
                    self._cache_files[dir_path] = actual_files

    def save(self, file_path):
        with self._lock:
            if not self.exist_file(file_path):
                self.append_file(file_path)
            with open(file_path, 'wb') as f:
                pickle.dump(self._cache_files, f)

    def load(self, file_path, validate_on_load=True):
        with self._lock:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self._cache_files = pickle.load(f)
                if validate_on_load:
                    cache_dirs = list(self._cache_files)
                    print('validating files')
                    for cache_dir in tqdm(cache_dirs, ascii=True):
                        self.validate_dir(cache_dir)
                    print('done')


global_file_cache = FileCacher()


class Cacher:
    def __init__(self, path):
        self._path = path
        # create dir if not exists
        create_dir(self._path)

    def __getitem__(self, item):
        if type(item) != str:
            item = str(item)
        path = os.path.join(self._path, calc_md5(item))
        if not global_file_cache.exist_file(path):
            raise KeyError('Item not exists')
        with open(path, 'rb') as f:
            return f.read()

    def __setitem__(self, key, value):
        if type(key) != str:
            key = str(key)
        path = os.path.join(self._path, calc_md5(key))
        if type(value) == str:
            value = bytes(value, 'utf8')
        elif type(value) != bytes:
            raise TypeError('value should be string or bytes')
        with open(path, 'wb') as f:
            f.write(value)
            global_file_cache.append_file(path)

    def get(self, item, default_item=None):
        try:
            return self.__getitem__(item)
        except KeyError:
            return default_item


class Crawler:
    def __init__(self, save_path_=None, cache_path_=None, nums_thread=5, begin_date=None,
                 max_page=2, max_buffer_size=3000):
        self._num_threads = nums_thread
        self._main_thd = None
        self._main_thd_started = threading.Event()
        self._fetch_finished = None
        self._max_page = max_page
        if begin_date is None or type(begin_date) != datetime:
            begin_date = datetime.fromordinal(datetime.now().date().toordinal()) - timedelta(days=2)
        self._date = begin_date
        self._page = 1
        if not save_path_:
            save_path_ = save_path
        self._save_path = save_path_
        self._cache = Cacher(cache_path_ if cache_path_ else cache_path)
        # handling abort event
        self._abort_event = threading.Event()
        self._abort_wait = []

        # handling variable buffer for main thread
        self._buffer_data = []
        self._buffer_lock = threading.RLock()
        self._buffer_empty = threading.Event()  # an event telling main thread to fetch more data
        self._buffer_empty.set()
        self._max_buffer_size = max_buffer_size

        # creating directory
        for i in range(1000):
            dst_path = os.path.join(save_path, str(i))
            create_dir(dst_path)
            if not global_file_cache.exist_dir_in_cache(dst_path):
                global_file_cache.add_cache_dir(dst_path)

    def _main_thd_cb(self):
        self._abort_wait = []
        self._abort_event.clear()
        self._fetch_finished = False

        try:
            # fetch ranking page
            print('Fetching ranking page (html mode)')
            # external loop for handling retrying
            while not self._abort_event.is_set():
                suc = False
                req = None
                while not suc:
                    if self._abort_event.is_set():
                        return
                    try:
                        req = requests.get('https://www.pixiv.net/ranking.php?mode=daily',
                                           headers={'User-Agent': ua}, proxies=proxy, timeout=15)
                        suc = True
                    except Exception as ex:
                        warn(str(ex))
                rep = req.content.decode('utf8')
                # handling non-200
                if req.status_code != 200:
                    print('HTTP Get failed with response code %d, retry in 0.5s' % req.status_code)
                    # wait 0.5s
                    if self._abort_event.wait(0.5):
                        break
                # parse tt
                pattern = re.compile(r'pixiv\.context\.token\s*=\s*"(?P<tt>\w+)";')
                match_result = re.finditer(pattern, rep)
                try:
                    match_result = next(match_result)
                except StopIteration:
                    match_result = None
                if not match_result:
                    print('Could not get tt from html, exited')
                    self._main_thd_started.set()
                    return
                self._tt = match_result.group('tt')
                break
            print('Got tt = "%s"' % self._tt)

            # starting parallel download thread here
            for _ in range(self._num_threads):
                event_to_wait = threading.Event()
                self._abort_wait.append(event_to_wait)
                worker = threading.Thread(target=self._worker_thd_cb, args=(event_to_wait,))
                worker.start()
            self._main_thd_started.set()

            headers = {'X-Requested-With': 'XMLHttpRequest',
                       'Referer': 'https://www.pixiv.net/ranking.php?mode=daily'}
            while self._buffer_empty.wait():
                if self._abort_event.is_set():
                    break

                # fetch from cacher
                key = '%s-p%d' % (self._date.strftime('%Y%m%d'), self._page)
                result = self._cache.get(key)
                if not result:
                    with self._buffer_lock:
                        print('Fetching ranking page(json mode), date=%s, page=%d, buffer=%d/%d' %
                              (str(self._date.date()), self._page, len(self._buffer_data), self._max_buffer_size))
                    params = {'mode': 'daily', 'date': self._date.strftime('%Y%m%d'), 'p': self._page,
                              'format': 'json', 'tt': self._tt}
                    suc = False
                    req = None
                    while not suc:
                        if self._abort_event.is_set():
                            return
                        try:
                            req = requests.get('https://www.pixiv.net/ranking.php', params=params, headers=headers,
                                               proxies=proxy, timeout=15)
                            suc = True
                        except Exception as ex:
                            warn(str(ex))
                    rep = req.content.decode('utf8')
                    # terminated state
                    if req.status_code == 404:
                        break
                    # append to cacher
                    self._cache[key] = rep
                    result = rep
                else:
                    result = result.decode('utf8')

                json_data = json.loads(result)
                buffer_data = self._parse_data(json_data)

                # append to buffer
                with self._buffer_lock:
                    self._buffer_data += buffer_data
                    # check buffer size
                    if len(self._buffer_data) >= self._max_buffer_size:
                        self._buffer_empty.clear()

                # next page
                self._page += 1

                if self._page > self._max_page:
                    self._page = 1
                    self._date -= timedelta(days=1)

        finally:
            print('main thd exited')
            self._fetch_finished = True
            for item in self._abort_wait:
                item.wait()

    def _parse_data(self, data):
        ret_data = []
        if data.get('contents', None):
            contents = data['contents']
            for content in contents:
                url = content['url']
                ranking_date = self._date
                ranking_page = self._page
                illust_id = int(content['illust_id'])
                illust_page_count = int(content['illust_page_count'])
                for page in range(illust_page_count):
                    single_illust_url = self._replace_url(url, page)
                    ret_data.append({'date': ranking_date, 'page': ranking_page,
                                     'illust_id': illust_id, 'illust_page': page,
                                     'url': single_illust_url})
        return ret_data

    @staticmethod
    def _replace_url(url, page):
        url_pattern = re.compile(r'(?P<schemas>https?)://(?P<host>([^./]+\.)+[^./]+)(/c/\d+x\d+)?'
                                 r'(?P<path_prefix>/img-master/img(/\d+){6}/\d+_p)\d+'
                                 r'(?P<path_postfix>_(master|square)\d+\.(jpg|png)).*')
        match = re.match(url_pattern, url)
        if match:
            schemas = match.group('schemas')
            host = match.group('host')
            path_prefix = match.group('path_prefix')
            path_postfix = match.group('path_postfix')
            return '%s://%s%s%d%s' % (schemas, host, path_prefix, page, path_postfix)
        url_pattern = re.compile(r'(?P<schemas>https?)://(?P<host>([^./]+\.)+[^./]+)(/c/\d+x\d+)?'
                                 r'(?P<path_prefix>/img-master/img(/\d+){6}/\d+)'
                                 r'(?P<path_postfix>_(master|square)\d+\.(jpg|png)).*')
        match = re.match(url_pattern, url)
        if match:
            schemas = match.group('schemas')
            host = match.group('host')
            path_prefix = match.group('path_prefix')
            path_postfix = match.group('path_postfix')
            if page != 0:
                warn('A non-pageable image url detected, your page should be 0 constantly, but got %d' % page)
            return '%s://%s%s%s' % (schemas, host, path_prefix, path_postfix)

        raise ValueError('The url "%s" could not match any replacement rules' % url)

    def _worker_thd_cb(self, thd_wait_event):
        try:
            while not self._abort_event.is_set():
                buffer_item = None
                with self._buffer_lock:
                    if len(self._buffer_data) > 0:
                        buffer_item = self._buffer_data[0]
                        self._buffer_data = self._buffer_data[1:]
                    if len(self._buffer_data) < self._max_buffer_size:
                        self._buffer_empty.set()

                # fetch failed, wait more time
                if buffer_item is None:
                    if self._fetch_finished or self._abort_event.wait(0.1):
                        break
                    continue

                # unpacking value
                date = buffer_item['date']
                page = buffer_item['page']
                illust_id = buffer_item['illust_id']
                illust_page = buffer_item['illust_page']
                url = buffer_item['url']

                # download file here
                dst_path = os.path.join(save_path, str(illust_id % 1000), '%dp%d.jpg' % (illust_id, illust_page))
                if not global_file_cache.exist_file(dst_path):
                    print('Downloading [%s #%d] [%d p%d] %s' % (date.strftime('%Y%m%d'), page,
                                                                illust_id, illust_page, url))
                    suc = False
                    while not suc:
                        try:
                            req = requests.get(url, headers={'Referer': 'https://www.pixiv.net/member_illust.php'
                                                                        '?mode=medium&illust_id=%d' % illust_id},
                                               timeout=15)
                            if req.status_code != 200:
                                warn('Error while downloading %d p%d : HTTP %d' %
                                     (illust_id, illust_page, req.status_code))
                                break

                            image = req.content
                            with open(dst_path, 'wb') as f:
                                f.write(image)
                            global_file_cache.append_file(dst_path)

                            suc = True
                        except Exception as ex:
                            print(ex)
        finally:
            thd_wait_event.set()
            print('thd exited')

    def start(self):
        self.abort()
        self._main_thd = threading.Thread(target=self._main_thd_cb)
        self._main_thd.start()

    def abort(self):
        self._abort_event.set()
        self.wait()

    def wait(self):
        if self._main_thd:
            self._main_thd_started.wait()
        for item in self._abort_wait:
            item.wait()


class DatabaseGenerator:
    # flags for illust_content_type
    ILLUST_CONTENT_TYPE_SEXUAL = 1
    ILLUST_CONTENT_TYPE_LO = 2
    ILLUST_CONTENT_TYPE_GROTESQUE = 4
    ILLUST_CONTENT_TYPE_VIOLENT = 8
    ILLUST_CONTENT_TYPE_HOMOSEXUAL = 16
    ILLUST_CONTENT_TYPE_DRUG = 32
    ILLUST_CONTENT_TYPE_THOUGHTS = 64
    ILLUST_CONTENT_TYPE_ANTISOCIAL = 128
    ILLUST_CONTENT_TYPE_RELIGION = 256
    ILLUST_CONTENT_TYPE_ORIGINAL = 512
    ILLUST_CONTENT_TYPE_FURRY = 1024
    ILLUST_CONTENT_TYPE_BL = 2048
    ILLUST_CONTENT_TYPE_YURI = 4096

    def __init__(self, path_to_save=None, cacher_path=None, max_page=2):
        self._cacher = Cacher(cacher_path if cacher_path else cache_path)
        if not path_to_save:
            path_to_save = db_path
        with open(path_to_save, 'w'):
            pass
        self._conn = sqlite3.connect(path_to_save)
        self._cursor = self._conn.cursor()
        self._max_page = max_page

        self._initialize()
        self._user_id_set = set()
        self._tag_id_dict = dict()
        self._rank_set = set()
        self._illust_id_set = set()
        self._illust_series_id_set = set()

    def _initialize(self):
        # initialize tables
        csr = self._cursor
        csr.execute("create table user (user_id bigint primary key, user_name varchar(255) not null,"
                    "profile_img varchar(255) not null)")
        csr.execute("create table illust_series (illust_series_id integer primary key, "
                    "illust_series_user_id bigint not null, illust_series_title varchar(255) not null,"
                    "illust_series_caption text(16383), illust_series_content_count integer not null,"
                    "illust_series_create_datetime datetime not null, page_url varchar(255) not null,"
                    "foreign key (illust_series_user_id) references user(user_id))")
        csr.execute("create table illust (title varchar(255), date datetime, url varchar(255), illust_type integer,"
                    "illust_book_style integer, illust_page_count integer, illust_content_type integer not null, "
                    "illust_series_id integer, illust_id bigint primary key, width integer not null, "
                    "height integer not null, user_id bigint not null, rating_count integer not null, "
                    "view_count integer not null, illust_upload_timestamp datetime not null, attr varchar(255),"
                    "foreign key (user_id) references user, foreign key (illust_series_id) references illust_series)")
        csr.execute("create table tag (tag_id integer primary key autoincrement, name varchar(255) not null unique)")
        csr.execute("create table illust_tags (illust_id bigint not null, tag_id integer not null,"
                    "foreign key (illust_id) references illust, foreign key (tag_id) references tag)")
        csr.execute("create table illust_rank (illust_id bigint not null, date datetime not null, "
                    "rank integer not null, yes_rank integer not null, foreign key (illust_id) references illust)")
        # indices to accelerate date-based query
        csr.execute("create index illust_date on illust(date)")
        csr.execute("create index illust_rank_date on illust_rank(date, rank)")
        self._conn.commit()

    def start(self):
        cur_date = datetime.now().date() - timedelta(days=2)
        cur_page = 1

        key = '%s-p%d' % (cur_date.strftime('%Y%m%d'), cur_page)
        data = self._cacher.get(key)
        while data:
            data = data.decode("utf8")
            # print('Parsing %s' % key)
            json_data = json.loads(data)
            try:
                contents = json_data['contents']
            except KeyError:
                break

            for item in contents:
                self._parse(item, cur_date)

            # next
            cur_page += 1
            if cur_page > self._max_page:
                cur_date -= timedelta(days=1)
                cur_page = 1
            key = '%s-p%d' % (cur_date.strftime('%Y%m%d'), cur_page)
            data = self._cacher.get(key)

        self._conn.commit()

    def _parse(self, json_obj, ranking_date):
        title = json_obj['title']
        date = json_obj['date']
        tags = json_obj['tags']
        url = json_obj['url']
        illust_type = json_obj['illust_type']
        illust_book_style = json_obj['illust_book_style']
        illust_page_count = json_obj['illust_page_count']
        user_name = json_obj['user_name']
        profile_img = json_obj['profile_img']
        illust_content_type = json_obj['illust_content_type']
        illust_series = json_obj['illust_series']
        illust_id = json_obj['illust_id']
        width = json_obj['width']
        height = json_obj['height']
        user_id = json_obj['user_id']
        rank = json_obj['rank']
        # hint: yes_rank is not YES! rank!, it's just the rank of yesterday, don't be treated XD
        yes_rank = json_obj['yes_rank']
        rating_count = json_obj['rating_count']
        view_count = json_obj['view_count']
        illust_upload_timestamp = json_obj['illust_upload_timestamp']
        attr = json_obj['attr']
        # converting illust_content_type
        flag_illust_content_type = 0
        if illust_content_type['sexual'] != 0:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_SEXUAL
        if illust_content_type['lo']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_LO
        if illust_content_type['grotesque']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_GROTESQUE
        if illust_content_type['violent']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_VIOLENT
        if illust_content_type['homosexual']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_HOMOSEXUAL
        if illust_content_type['drug']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_DRUG
        if illust_content_type['thoughts']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_THOUGHTS
        if illust_content_type['antisocial']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_ANTISOCIAL
        if illust_content_type['religion']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_RELIGION
        if illust_content_type['original']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_ORIGINAL
        if illust_content_type['furry']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_FURRY
        if illust_content_type['bl']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_BL
        if illust_content_type['yuri']:
            flag_illust_content_type |= self.ILLUST_CONTENT_TYPE_YURI
        # querying user data
        csr = self._cursor
        if not self._user_id_set.issuperset({user_id}):
            csr.execute("insert into user(user_id, user_name, profile_img) values (?, ?, ?)",
                        (user_id, user_name, profile_img))
            self._user_id_set.add(user_id)
        # handling illust_series
        if type(illust_series) != bool:
            illust_series_id = illust_series['illust_series_id']
            illust_series_user_id = illust_series['illust_series_user_id']
            illust_series_title = illust_series['illust_series_title']
            illust_series_caption = illust_series['illust_series_caption']
            illust_series_content_count = illust_series['illust_series_content_count']
            illust_series_create_datetime = illust_series['illust_series_create_datetime']
            page_url = illust_series['page_url']
            if not self._illust_series_id_set.issuperset({illust_series_id}):
                csr.execute("insert into illust_series(illust_series_id, illust_series_user_id, "
                            "illust_series_title, illust_series_caption, illust_series_content_count, "
                            "illust_series_create_datetime, page_url) values (?, ?, ?, ?, ?, ?, ?)",
                            (illust_series_id, illust_series_user_id, illust_series_title, illust_series_caption,
                             illust_series_content_count, illust_series_create_datetime, page_url))
                self._illust_series_id_set.add(illust_series_id)
            illust_series = illust_series_id
        else:
            illust_series = None
        # tags
        for tag in tags:
            if self._tag_id_dict.get(tag, None):
                tag_id = self._tag_id_dict[tag]
            else:
                csr.execute("insert into tag(name) values (?)", (tag,))
                tag_id = len(self._tag_id_dict) + 1
                self._tag_id_dict[tag] = tag_id
            csr.execute("insert into illust_tags(illust_id, tag_id) values (?, ?)", (illust_id, tag_id))
        # converting date
        reg_ptn = re.compile('(\\d+)年(\\d+)月(\\d+)日\\s(\\d+):(\\d+)')
        match = re.match(reg_ptn, date)
        if match:
            date_year, date_month, date_day, date_hour, date_minute = (int(match.group(x)) for x in range(1, 6))
            date = datetime(date_year, date_month, date_day, date_hour, date_minute)
        illust_upload_timestamp = datetime.fromtimestamp(illust_upload_timestamp)
        if not self._illust_id_set.issuperset({illust_id}):
            csr.execute("insert into illust(title, date, url, illust_type, illust_book_style, illust_page_count, "
                        "illust_content_type, illust_series_id, illust_id, width, height, user_id, rating_count, "
                        "view_count, illust_upload_timestamp, attr) "
                        "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (title, date, url, illust_type, illust_book_style, illust_page_count, flag_illust_content_type,
                         illust_series, illust_id, width, height, user_id, rating_count, view_count,
                         illust_upload_timestamp, attr))
            self._illust_id_set.add(illust_id)

        if not self._rank_set.issuperset((illust_id, ranking_date, rank, yes_rank)):
            csr.execute("insert into illust_rank(illust_id, date, rank, yes_rank) values (?, ?, ?, ?)",
                        (illust_id, ranking_date, rank, yes_rank))
            self._rank_set.add((illust_id, ranking_date, rank, yes_rank))


if __name__ == '__main__':
    global_file_cache.load(os.path.join(cache_path, 'index'))
    print('Crawler starting')
    a = Crawler()
    a.start()
    a.wait()
    global_file_cache.save(os.path.join(cache_path, 'index'))
    print('Database generator starting')
    a = DatabaseGenerator()
    a.start()
    global_file_cache.save(os.path.join(cache_path, 'index'))

```

### 后日谈

给这个爬虫爬到的图片标了下数据，自己跑了个faster-rcnn的[动漫脸识别]，结合[动漫脸识别]里面提到的基于opencv的识别方法，可以把opencv中CascadeClassifier那惨不忍睹的识别结果过滤到几乎为100%正确的结果，这些几乎没有错误的结果可以用来实验一下各种的GAN。

CascadeClassifier的优点就是检测出来的脸型比较单一，当然这也是它的缺点所在，就因为这一点，过滤掉了一大半的结果（心痛）。

![12](https://zhouxuebin.club:4433/data/2019/01/pixiv_crawler_img_12.png)

后续会放出处理faster-rcnn和CascadeClassifier生成的结果，并且对两个结果求IoU比例、根据IoU进行边框匹配并裁剪缩放的过程及代码。

[p站日榜]: https://www.pixiv.net/ranking.php?mode=daily
[动漫脸识别]: https://github.com/qhgz2013/anime-face-detector/