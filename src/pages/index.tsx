import React from "react"
import { PageProps, Link } from "gatsby"

import Layout from "../layouts/layout"

const IndexPage: React.FC<PageProps> = () => {
    return (
        <Layout title="資料置き場">
            <h1 className="uk-heading-primary">資料置き場</h1>
            <ul>
                <li>
                    <Link to="/prml-10">
                        パターン認識と機械学習 第10章 近似推論法
                    </Link>
                </li>
                <li>
                    <Link to="/prml-11">
                        パターン認識と機械学習 第11章 サンプリング法
                    </Link>
                </li>
            </ul>
        </Layout>
    )
}

export default IndexPage
