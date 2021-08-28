import React from "react"
import { Helmet } from "react-helmet"

import Header from "./header"

interface LayoutProps {
    title: string
}

const Layout: React.FC<LayoutProps> = ({ title = "", children }) => {
    const returnTop = () => {
        window.scrollTo({
            top: 0,
            behavior: "smooth",
        })
    }
    return (
        <>
            <Helmet
                title={title}
                meta={[
                    {
                        name: "viewport",
                        content: "width=device-width, initial-scale=1",
                    },
                    {
                        name: "robots",
                        content: "noindex",
                    },
                ]}
            >
                <link
                    rel="stylesheet"
                    href="https://cdn.jsdelivr.net/npm/uikit@3.7.2/dist/css/uikit.min.css"
                />
                <script src="https://cdn.jsdelivr.net/npm/uikit@3.7.2/dist/js/uikit.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/uikit@3.7.2/dist/js/uikit-icons.min.js"></script>
            </Helmet>
            <Header />
            <div className="uk-margin-top uk-margin-bottom uk-margin-left uk-margin-right">
                {children}
                <button
                    className="uk-icon-button uk-position-fixed uk-position-medium uk-position-bottom-right"
                    uk-icon="chevron-up"
                    style={{ zIndex: 2000, opacity: "0.8" }}
                    onClick={returnTop}
                />
            </div>
            <footer className="uk-background-muted uk-padding-small">
                <div className="uk-text-center">Hosted by Vercel</div>
            </footer>
        </>
    )
}

export default Layout
