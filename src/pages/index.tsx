import React from "react"
import { PageProps, Link } from "gatsby"

import Layout from "../layouts/layout"

const IndexPage: React.FC<PageProps> = () => {
    return (
        <Layout title="資料置き場">
            <div uk-height-viewport="expand: true">
                <h1 className="uk-heading-primary">資料置き場</h1>
                <ul>
                    <li>
                        <span>現代数理統計学の基礎</span>
                        <ul>
                            <li>
                                <Link to="/stats-04">
                                    現代数理統計学の基礎 04 多次元確率変数の分布
                                </Link>
                            </li>
                            <li>
                                <Link to="/stats-06">
                                    現代数理統計学の基礎 06 標本分布とその近似
                                    (6.3,6.4)
                                </Link>
                            </li>
                        </ul>
                    </li>
                    <li>
                        <span>例解 ディジタル信号処理入門</span>
                        <ul>
                            <li>
                                <Link to="/signal-process-05">
                                    信号処理 5章 伝達関数
                                </Link>
                            </li>
                            <li>
                                <Link to="/signal-process-08">
                                    信号処理 8章
                                    離散時間システムの周波数領域表現
                                </Link>
                            </li>
                        </ul>
                    </li>
                    <li>
                        <span>パターン認識と機械学習</span>
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
                    </li>
                </ul>
            </div>
        </Layout>
    )
}

export default IndexPage
