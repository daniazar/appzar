---
title: Issues when using configuration files in Scaling environments
excerpt: >-
  Dealing with issues on Azure Blob Storage throttling, Redis cache failover, and exponential retry policies in multi-region scaled environments.
date: '2020-03-09'
tags: ['MoltenMinds', 'Azure', 'Cloud', 'C#', 'Architecture', 'Scaling']
thumb_img_path: /images/scale.png
content_img_path: /images/scale.png
template: post
---

Hello,

We will continue our last post on problems that appear when scaling. This time we want to talk about what happens when you need to use configuration parameters in an environment. These values can be stored in many different places. In the case of the project we had, they were stored in [**Azure Blob Storage**](https://azure.microsoft.com/en-us/services/storage/blobs/). The problem in this case appeared when enabling auto scaling.

## About the project

First we need to explain the project. It was a complex backend application in **C#**. The architecture of the project was designed with scaling in mind. It consisted of Frontend and Backend App Service Servers. Each server contained many WebJobs. The different WebJobs needed several parameters to run correctly. When they were launched each one needed to read a configuration file containing these parameters.

We decided to use configuration files instead of saving the values directly to the database, first because of a performance issue, but also to be able to have different sets of parameters for each region and environment. These values could be changed from the frontend so each job had to refresh them after some time so the changes propagate across the app in a similar way that DNS changes propagate.

To read the files we had three checks: first we search in local memory cache for the file. Then we search in [**Redis**](https://azure.microsoft.com/en-us/services/cache/) — if not present we look for the file in [**Azure Blob Storage**](https://azure.microsoft.com/en-us/services/storage/blobs/).

With this architecture the program was already running in multiple regions with multiple servers. And in each region it was configured with App Service using a considerable number of instances.

## The Problem

The Application was working as expected. And a new objective was defined to reduce the costs. After solving the scaling problems we started to find a couple times during two-week periods where the program was logging errors when accessing the configuration files.

At first we thought there was a problem with Azure Redis and Blob Storage both at the same time, but the issue happened again after some time and that couldn't be a coincidence. Then we started to investigate the possible cause of the issue.

## The solution

After days of reading what could cause these reading errors when accessing the configuration files, the first thing we could correlate was our errors with the downtimes of the Redis cache service. So we now knew that each server could have at most 20 instances, and these instances had more than 30 jobs, so we had at most 600 processes running a request asking for the file. That is if every instance was started at the same time. So why would the Blob Storage fail for 600 file requests?

We found the [**documentation**](https://docs.microsoft.com/en-us/azure/storage/blobs/scalability-targets) on scalability and performance for Blob storage. There we can see that there is a throttle limit independent of the Storage tier you have. This limit is 500 requests per second, which we were hitting sometimes; even with our retry policy we were having errors.

The retry policy was waiting for a constant time; upon changing it to an exponential backoff policy, most of the issues were resolved.

Then we read about the [**naming convention**](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-performance-checklist#partitioning) to improve Storage Service load balancing. We were saving in the same Storage account all the queue messages and leasing files without appropriate naming. Upon changing that we resolved all the issues with the blob services.

Hope you learnt something about managing Blob files in a scaling application.

Regards.