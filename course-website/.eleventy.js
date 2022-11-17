const markdownIt = require("markdown-it");

module.exports = function (eleventyConfig) {
  eleventyConfig.addPassthroughCopy("./src/css");
  eleventyConfig.addWatchTarget("./src/css/");

  eleventyConfig.addPassthroughCopy("./src/fonts");
  eleventyConfig.addWatchTarget("./src/fonts");

  eleventyConfig.addPassthroughCopy("./src/img");
  eleventyConfig.addWatchTarget("./src/img");

  eleventyConfig.addPassthroughCopy("./src/favicon.ico");
  eleventyConfig.addWatchTarget("./src/favicon.ico");

  let markdownItOptions = {
    html: true,
    breaks: true,
  };
  
  eleventyConfig.setLibrary("md", markdownIt(markdownItOptions));

  return {
    dir: {
      input: "src",
      output: "public",
    },
  };
};
