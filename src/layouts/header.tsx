import React from "react"
import { Link } from "gatsby"

const Header: React.FC = () => {
    return (
        <nav className="uk-navbar uk-navbar-container uk-padding-small">
            <div className="uk-navbar-left">
                <Link
                    to="/"
                    className="uk-navbar-brand uk-icon-button uk-hidden-small"
                    uk-icon="icon: home; ratio: 1.5"
                />
            </div>
            <div className="uk-navbar-right">
                <a
                    href="/"
                    className="uk-icon-button uk-hidden-small"
                    uk-icon="icon: github; ratio: 1.5"
                />
            </div>
        </nav>
    )
}

export default Header
