---
title: Easily create a SSL certificate
excerpt: >-
  When developing an app you will need to use HTTPS more frequently as browser are blocking simple http requests and some tools stop working without a correct certificate. Generate your certificate easy with this guide.
date: '2018-08-22'
thumb_img_path: images/SSL-Certificates.jpg
content_img_path: images/SSL-Certificates.jpg
template: post
---

> Success is no accident. It is hard work, perseverance, learning, studying, sacrifice and most of all, love of what you are doing or learning to do. - Pele

# Generate a Trusted SSL Certificate

When developing an app you will need to use HTTPS more frequently as browser are blocking simple http requests and some tools stop working without a correct certificate. Also it is important when using Docker and micro services with your own registry.

Because of this you need to learn how to generate a certificate. The easiest option is to use OpenSSL to generate your own certificate, but its interface can be cumbersome. To streamline the use of it and not having to read documentation each time you can clone the following repository and invoke the script. It will generate a Certificate based on a configuration file with easy to understand values.

```
git clone https://github.com/daniazar/generate-trusted-ssl-certificate.git
cd <PATH_TO_REPO_FOLDER>/generate-trusted-ssl-certificate
bash generate.sh
```

## Requirements
Install OpenSsl on your OS

# OS X
You should be all set. OS X has by default openssl installed.

# Windows 10
Install openssl. I recommend using the Git bash. It has openssl preinstalled. Git Bash is bundled with the Git installer.

https://git-scm.com/

## Configuration

You can adjust the `[dn]` part of the `openssl-custom.cnf` file to whatever you prefer.

```
[dn]
C = <COUNTRY>
ST = <STATE>
L = <LOCALITY / CITY>
O = <ORGANIZATION>
OU = <ORGANIZATION_UNIT>
emailAddress = <EMAIL_ADDRESS>
CN = <HOSTNAME / IP_ADDRESS>
```

## Angular
In case of angular if you need to run ng serve with a secure port on development. This is the case for example if your application contains an iframe of a website with https.

To run angular cli with https use

```
ng serve --ssl true --ssl-cert server.crt --ssl-key server.key
```
