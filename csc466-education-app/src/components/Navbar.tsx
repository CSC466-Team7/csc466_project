import React from "react";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import { AppBar, Toolbar, Link } from "@material-ui/core";

// TODO: collapse links into hamburger menu on small screen
const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    bar: {
      justifyContent: "space-between",
    },
    links: {
      color: "white",
      padding: "0 12px",
    },
  }),
);

export default function ButtonAppBar() {
  const classes = useStyles();

  return (
    <AppBar position="static">
      <Toolbar className={classes.bar}>
        <h3>
          Decision Trees
        </h3>
        <span>
          <Link className={classes.links} href="/#/">
            Home
          </Link>
          <Link className={classes.links} href="/#/introduction">
            Introduction
          </Link>
          <Link className={classes.links} href="/#/example">
            Examples
          </Link>
          <Link className={classes.links} href="/#/preliminary-skills">
            Preliminary Skills
          </Link>
        </span>
      </Toolbar>
    </AppBar>
  );
}
