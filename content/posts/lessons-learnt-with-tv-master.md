---
title: Lessons Learnt with TV Master
excerpt: >-
  Problems and solutions found while making a tv streaming site.
date: '2020-05-25'
thumb_img_path: images/cors.png
content_img_path: images/cors.png
template: post
---

> CORS and security measures cause a lot of trouble in not common scenarios.

Hi, this is a continuation of the last TV Master post. To explain most of the issues and solutions I encountered upon making the TV Master site.

The first issue i had was a security issue with the **iframe**. The problem appears when adding an HTTPS iframe to an HTTP website or vice versa. In one case the browser refuse to load it, while in the other it prompts the user to allow it. The prompt is not easily visible so its easy to miss. Also on mobile browsers there is no prompt.

![example unsafe iframe](/images/unsafe.jpg)

The solution is to use HTTPS on your site with the appropriate certificate. if you need help making a certificate read the post title **title: Easily create a SSL certificate**. The other way around is blocked automatically by most browsers.

Then we go into iframes permissions. you could enable or block popups, javascript, redirects, etc. The implementation of this appears to differ for each major browser. I would like to block redirects popups to block Ads but need to leave javascript as most videos streams don't use the HTML5 video tag. The best browser handling this issues appear to be **Opera**.

We begin with CORS.
![example unsafe iframe](/images/cors2.png)
With CORS on the server side you can enable or block other sites from using your resources. this could be done by IP, Origins Domain, etc. As times go by more providers block their signals so they cant be played on other websites. This is really cumbersome as we don't have control over the browser origin url arguments on requests. if we had we could do easily Cross Site Scripting hacks. There are two ways to bypass this, having a backend server acting as proxy or not using a browser as the frontend. But the app was already started and the architecture and languages chosen.

Then we used electron. An Electron app is a web engine encapsulated website with access to the relaying OS. Electron by default doesn't help us violating web security. but with some tinkering lets us bypass CORS by setting origin URL per request as whatever we want. In Electron we need to change the iFrame tags to WebViews tags for security to prevent the iframe from accessing the relaying OS features. So the web and Desktop implementation needs to differ with the App having better functionality.

I think that covers the big issues presented while developing this App.
