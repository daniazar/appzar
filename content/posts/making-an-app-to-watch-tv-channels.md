---
title: Making an App to watch TV Channels
excerpt: >-
  Why I decided to make my own tv streaming site, and how i achieve it.
date: '2018-08-20'
thumb_img_path: images/tv-master.png
content_img_path: images/tv-master.png
template: post
---

> I grow tired of small rectangular streaming on sites and ads.

Hello guys, first a little about me. I don't like to hear music while working but i like having news, a movie or a tv show in the background. But with some channels I used in the past starting to add annoying Ads or timed refresh of the website, I needed a change. After looking around and not finding what i was looking. I decided it was time to make my own website to stream content. Just look at the following example:

![example of little video in news site](/images/america-tv.png)

> **Use Opera Web Browser** for TV Master to work best.

I decided to make the App. It started as a Web Application, using typescript and angular. The idea was to make it easy to watch and change channels. Make the video size nearly full site with the option to go Fullscreen. For the backend it uses firebase, for the database and the login. Admin users can change and add new channels, while the rest can report non working channels. As i don't host the streams they change frequently and it makes some channels to stop working. Added tabs to enable multiple sources per channel and a Guide button for each channel. Then added a calendar feature for events like sports and which channel it was available. Some of the sources could be easily added like Youtube live streams but others where a bit shady origin with some complex iframe programming to be added. Also each had to be easily edited on real time.

Example of the video adjusting its size to the windows width. With a limit on height.
![video in small window](/images/tvmaster-small.png)

Example of the video on max height.
![video in big window](/images/tvmaster-full.png)

Example of changing the channel without interrupting the video in the main view.
![video in big window](/images/tvmaster-channelpopup.png)

I will make another post explaining the problems faced and the learnt things.
