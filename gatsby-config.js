"use strict"

require("ts-node").register({
    compilerOptions: {
        target: "esnext",
        module: "commonjs",
    },
})
require("./src/__generated__/gatsby-types")

const {
    onCreateNode,
    createPages,
} = require("./gatsby-node.ts")
exports.onCreateNode = onCreateNode 
exports.createPages = createPages

module.exports = {
    plugins: [
        "gatsby-plugin-typegen",
        "gatsby-transformer-sharp",
        "gatsby-plugin-sharp",
        {
            resolve: "gatsby-source-filesystem",
            options: {
                name: "docs",
                path: `${__dirname}/contents/docs`,
            },
        },
        {
            resolve: "gatsby-source-filesystem",
            options: {
                name: "images",
                path: `${__dirname}/contents/images`,
            },
        },
        {
            resolve: "gatsby-source-filesystem",
            options: {
                name: "videos",
                path: `${__dirname}/contents/videos`,
            },
        },
        {
            resolve: 'gatsby-plugin-copy-files',
            options: {
                source: `${__dirname}/contents/videos`,
                destination: '/videos'
            }
        },
        {
            resolve: "gatsby-transformer-remark",
            options: {
                plugins: [
                    {
                        resolve: "gatsby-remark-images",
                        options: {
                            maxWidth: 700,
                        },
                    },
                    {
                        resolve: "gatsby-remark-katex",
                        options: {
                            strict: false,
                        },
                    },
                ],
            }
        },
    ]
}
