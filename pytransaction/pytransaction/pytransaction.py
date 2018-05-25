import urllib.request
url="http://www.baidu.com"
headers=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36")
opener=urllib.request.build_opener()
opener.addheaders=[headers]
data=opener.open(url).read()
fhandle=open("D:/test/3.html","wb")
fhandle.write(data)
fhandle.close()
