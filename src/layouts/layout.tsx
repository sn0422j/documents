import React from "react"
import { Helmet } from "react-helmet"

import Header from "./header"

import "uikit/dist/css/uikit.min.css"

interface LayoutProps {
    title: string
}

const Layout: React.FC<LayoutProps> = ({ title = "", children }) => {
    if (typeof window !== "undefined") {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const UIkit = require("uikit/dist/js/uikit.min")
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const icons = require("uikit/dist/js/uikit-icons.min")
        UIkit.use(icons)
    }
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
                        name: "icon",
                        content: "favicon.ico",
                    },
                    {
                        name: "robots",
                        content: "noindex",
                    },
                ]}
            ></Helmet>
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
