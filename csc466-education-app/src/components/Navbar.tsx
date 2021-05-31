import { AppBar, Link, Toolbar } from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import React from "react";

// TODO: collapse links into hamburger menu on small screen
const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    heading: {
      color: "white"
    },
    bar: {
      justifyContent: "space-between"
    },
    links: {
      color: "white",
      padding: "0 12px"
    }
  })
);

export default function ButtonAppBar() {
  const classes = useStyles();
  
  return (
    <AppBar position="static">
      <Toolbar className={classes.bar}>
        <Link href="/#/">
          <h3 className={classes.heading}>
            Decision Trees
          </h3>
        </Link>
        <span>
          <Link className={classes.links} href="/#/introduction">
            Introduction
          </Link>
          <Link className={classes.links} href="/#/tutorials">
            Tutorials
          </Link>
          <Link className={classes.links} href="/#/preliminary-skills">
            Preliminary Skills
          </Link>
        </span>
      </Toolbar>
    </AppBar>
  );
}
