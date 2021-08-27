import { resolve } from "path"
import { createFilePath } from "gatsby-source-filesystem"
import type { GatsbyNode } from "gatsby"

export const createPages: GatsbyNode["createPages"] = async ({
    graphql,
    actions,
}) => {
    const { createPage } = actions

    const blogPost = resolve(`./src/templates/docs.tsx`)

    const result = await graphql<{
        allMarkdownRemark: Pick<GatsbyTypes.Query["allMarkdownRemark"], "nodes">
    }>(
        `
            query DocsQuery {
                allMarkdownRemark {
                    nodes {
                        fields {
                            slug
                        }
                        frontmatter {
                            title
                        }
                    }
                }
            }
        `
    )

    const posts = result!.data!.allMarkdownRemark!.nodes

    if (posts.length > 0) {
        posts.forEach((post) => {
            createPage({
                path: post!.fields!.slug!,
                component: blogPost,
                context: {
                    slug: post!.fields!.slug,
                },
            })
        })
    }
}

export const onCreateNode: GatsbyNode["onCreateNode"] = ({
    node,
    actions,
    getNode,
}) => {
    const { createNodeField } = actions

    if (node.internal.type === `MarkdownRemark`) {
        const value = createFilePath({ node, getNode })

        createNodeField({
            name: `slug`,
            node,
            value,
        })
    }
}
