@echo off 
setlocal enableextensions 
 
set Ya_LINK="https://yadi.sk/d/cxihXiIFh7EXW" 
 
wget --no-check-certificate -Oyadisk.html --keep-session-cookies --save-cookies=yadisk_cookies.txt %YA_LINK% 
for /f "tokens=*" %%s in ('type yadisk.html^|sed "/\""ckey\"":\""/!d;s/^.*\""ckey\"":\""//;s/\"".*$//"') do set Ya_CKEY=%%s 
for /f "tokens=*" %%s in ('type yadisk.html^|sed "/\""hash\"":\""/!d;s/^.*\""hash\"":\""//;s/\"".*$//"') do set Ya_HASH=%%s 
for /f "tokens=*" %%s in ('type yadisk.html^|sed "/\""twitter:title\""/!d;s/^.*\""twitter:title\""//;s/^[^""]*\""//;s/\"".*$//"') do set Ya_NAME=%%s 
del /f /q yadisk.html 
wget --no-check-certificate -O- --post-data="_ckey=%Ya_CKEY%&_name=getLinkFileDownload&hash=%Ya_HASH%" --load-cookies=yadisk_cookies.txt "https://disk.yandex.ru/handlers.jsx"|sed "/\""url\"":\""/!d;s/^.*\""url\"":\""//;s/\"".*$//"|wget --no-check-certificate -i- -O"%Ya_NAME%" 
del /f /q yadisk_cookies.txt 