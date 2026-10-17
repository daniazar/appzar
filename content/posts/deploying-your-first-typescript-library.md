---
title: Deploying my first typescript library
subtitle: >-
  How to make and deploy a typescript library to NPM.
excerpt: >-
  Writing modular code is a good idea. And nothing is more modular than writing a library. How can you write a TypeScript library? Well, that's exactly what this tutorial is about! This tutorial works with TypeScript 3.x

date: '2020-06-10'
thumb_img_path: images/typescript-library.png
content_img_path: images/typescript-library.png
template: post
---

Writing modular code is a good idea. And nothing is more modular than writing a library. How can you write a TypeScript library? Well, that's exactly what this tutorial is about! This tutorial works with TypeScript 3.x


In my case I wanted to play a little with L-System and fractals. To achieve this first we need a Turtle Graphics engine. I didnt found one for typescript so i made one and published it to NPM registry. I will assume you already know how to create a typescript project.


## Create an index.ts file
Add an index.ts file to your src folder. Its purpose is to export all the parts of the library you want to make available for consumers. In my case it would simply be:

```
export { Color } from "./Color";
export { Figure } from "./Figure";
export { Point } from "./Point";
export { Turtle } from "./Turtle";
```

## Configure the package.json
The package name is what the consumer is going to use to import functionality from your library later on. The whole package.json would look like so:

typescript-library/package.json
```
{
  "name": "turtle-ts",
  "version": "2.0.0",
  "description": "Turtle graphics engine in typescript",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "repository": {
    "type": "git",
    "url": "git+https://github.com/daniazar/turtlets.git"
  },
  "keywords": [
    "turtle",
    "engine",
    "typescript"
  ],
  "author": "Daniel Azar",
  "license": "MIT",
  "bugs": {
    "url": "https://github.com/daniazar/turtlets/issues"
  },
  "homepage": "https://github.com/daniazar/turtlets#readme",
  "dependencies": {
    "eslint": "^7.2.0"
  }
}
```

The package name you choose will be used by people for their imports, so choose wisely!

There's also one all-important flag in this package.json: You have to declare where to find the type declarations! This is done using "types": "dist/index.d.ts" Otherwise the consumer won't find your module!

## Configure .npmignore
If you would compile your code now with tsc and run npm publish, you'd publish the dist folder along with all the other files. However, you don't really need the source folder. To avoid publishing unnecessary files to npm, you can create an .npmignore file. In our case it's simple:

```
tsconfig.json
src
```
As your project grows, you will need to ignore more files, for example test files.

Step 6: Publish to npm
To publish your first version to npm run:

```
tsc
npm init
npm publish
```

Now you're all set to go! Consume the library anywhere you want. in this case by running:

```
npm install --save turtle-ts
```

For subsequent releases, use the semvar principle. When you make a patch / bugfix to your library, you can run npm version patch, for new features run npm version minor and on breaking changes of your api run npm version major.