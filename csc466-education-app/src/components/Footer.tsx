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
    },
  }),
);

export default function Footer() {
  const classes = useStyles();

  return (
    <AppBar position="static" component="footer" className={classes.root}>
      <Grid container spacing={2} className={classes.grid}>
        <Grid item xs={4}>
          <h2>Decision Trees</h2>
          <p>CSC 466 – Spring 2021</p>
          <p>Cal Poly – Dr. Anderson</p>
        </Grid>

        <Grid item xs={2}>
          <h3>
            <Link to='/introduction'>Introduction</Link>
          </h3>
          <Link to='/decision-trees'>What are Decision Trees</Link>
          <Link to='/getting-started'>Getting Started</Link>
        </Grid>

        <Grid item xs={2}>
          <h3>
            <Link to='/examples'>Examples</Link>
          </h3>
          <Link to='/example-1'>Example #1</Link>
          <Link to='/example-2'>Example #2</Link>
        </Grid>

        <Grid item xs={2}>
          <h3>
            <Link to='/preliminary-skills'>Preliminary Skills</Link>
          </h3>
          <Link to='/preliminary-skills/numpy'>Numpy</Link>
          <Link to='/preliminary-skills/pandas'>Pandas</Link>
          <Link to='/preliminary-skills/scikit-learn'>Sci-kit Leanr</Link>
        </Grid>

        <Grid item xs={2}>
          <h3>Resources</h3>
          <a
            href="https://github.com/CSC466-Team7/csc466_project"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
          <p>YouTube</p>
        </Grid>
      </Grid>
    </AppBar>
  );
}
