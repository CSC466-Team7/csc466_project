import React from "react";
import { Link } from "react-router-dom";
import { AppBar, Grid } from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      overflow: "hidden",
      padding: "40px 0",
    },
    grid: {
      display: "flex",
      flexWrap: "wrap",
      justifyContent: "space-between",
      width: "80%",
      margin: "0 auto",
      "& h3 a": {
        color: "white",
      },
      "& a": {
        display: "block",
        margin: "16px 0",
        color: "#ddd",
      },
      "& div": {
        display: "inline-block",
      },
    },
    siteTag: {
      width: "40%",
      minWidth: "300px",
      marginBottom: "30px",
    },
    linkTag: {
      width: "16.6%",
      minWidth: "200px",
      paddingRight: "3px",
    },
  }),
);

export default function Footer() {
  const classes = useStyles();

  return (
    <AppBar position="static" component="footer" className={classes.root}>
      <div className={classes.grid}>
        <div className={classes.siteTag}>
          <h2>Decision Trees</h2>
          <p>CSC 466 – Spring 2021<br />Cal Poly – Dr. Anderson</p>
        </div>

        <div className={classes.linkTag}>
          <h3>Introduction</h3>
          <Link to='/introduction'>Intro to Decision Trees</Link>
          <Link to='/getting-started'>Getting Started</Link>
        </div>

        <div className={classes.linkTag}>
          <h3>Tutorials</h3>
          <Link to='/example/1'>Example #1</Link>
          <Link to='/example/2'>Example #2</Link>
        </div>

        <div className={classes.linkTag}>
          <h3>Resources</h3>
          <Link to='/preliminary-skills'>Preliminary Skills</Link>
          <Link
            to="https://github.com/CSC466-Team7/csc466_project"
            target="_blank"
            rel="noreferrer noopener"
          >
            GitHub
          </Link>
        </div>

      </div>
    </AppBar>
  );
}
