import React from "react"
import { graphql } from "gatsby"
import "katex/dist/katex.min.css"
import "./docs.css"

import Layout from "../layouts/layout"

interface PageProps {
    data: GatsbyTypes.DocsPagesQueryQuery
}

const DocsPage: React.FC<PageProps> = ({ data }) => {
    const title: string = data.markdownRemark?.frontmatter?.title ?? ""
    return (
        <Layout title={title}>
            <h1 className="uk-heading-primary">{title}</h1>
            <div
                className="uk-text-break"
                dangerouslySetInnerHTML={{
                    __html: data.markdownRemark?.html ?? "",
                }}
            />
        </Layout>
    )
}

export default DocsPage

export const docsPagesQuery = graphql`
    query DocsPagesQuery($slug: String!) {
        markdownRemark(fields: { slug: { eq: $slug } }) {
            html
            frontmatter {
                title
            }
        }
    }
`
