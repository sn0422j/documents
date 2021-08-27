import React from "react"
import { graphql } from "gatsby"
import "katex/dist/katex.min.css"
import "./docs.css"

interface PageProps {
    data: GatsbyTypes.DocsPagesQueryQuery
}

const DocsPage: React.FC<PageProps> = ({ data }) => {
    return (
        <>
            <h1>{data.markdownRemark?.frontmatter?.title ?? ""}</h1>
            <div
                className="main-body"
                dangerouslySetInnerHTML={{
                    __html: data.markdownRemark?.html ?? "",
                }}
            />
        </>
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
