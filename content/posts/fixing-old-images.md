---
title: Fixing old images
subtitle: An interesting project during the Covid-19 pandemic
excerpt: >-
  During the COVID-19 pandemic we all have a lot more time at home that what we used to. How i applied some of that time on postponed tasks.
date: '2020-04-10'
thumb_img_path: images/metadata.png
content_img_path: images/metadata.png
template: post
---

During the COVID-19 pandemic we all have a lot more time at home that what we used to. This is due to most countries implementing a quarantine. To better use this time I decided to finish some family projects that were always postponed by other things. This time we are talking about digitalizing all the old photos inside our home.

For this project we used a Fujitsu ScanSnap iX500. once all the photo albums were scanned we found to big issues.

  1. Decoloration of old pictures. usually a red tint.
  2. if merged with google photos or other system all the pictures metadata is wrong.

## Decoloration
This will be the subject of a following post that includes an IA Neural Network to fix them.

## Metadata

An image file can contain several information regarding the time and mechanism when it was taken. We don't think about this data as most smartphones manage this information correctly now a days. But for old photos its a different paradigm.

Most photos software wont let you edit this metadata or just edit it once at a time. That can be an issue if you have as many photos as I do.

![Amount of pictures](/images/scanned-folder.png)

To fix this i searched for programs or ways to do it but all had complex interfaces or didn't work as expected on every file. So I decided to make my own application to fix them. The app ended being a Windows 10 application using WinUI 3.0 Alpha. because of the alpha status some of the common components like a map was not working.

The metadata that I wanted to fix the most was the date taken tag and the location of the picture. As I already have them separated by albums we could assume in most cases that all the pictures in an album were taken on the same period / location. So the app lets you choose one picture or a folder and process one at a time or all together.

You input the location, that gets converted using Bing Maps API for windows to coordinates. The location can be a full address or just a city or country.

![application running](/images/image-metadata.png)

If you want to check the application code it is located at my GitHub account.