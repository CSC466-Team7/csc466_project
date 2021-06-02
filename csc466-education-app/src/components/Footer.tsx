import { AppBar, Typography } from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import React from "react";
import { Link } from "react-router-dom";
import { tutorials } from "../data";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      overflow: "hidden",
      padding: "40px 0"
    },
    grid: {
      display: "flex",
      flexWrap: "wrap",
      justifyContent: "space-between",
      width: "80%",
      margin: "0 auto",
      "& h3 a": {
        color: "white"
      },
      "& a": {
        display: "block",
        margin: "16px 0",
        color: "#ddd"
      },
      "& div": {
        display: "inline-block"
      }
    },
    siteTag: {
      width: "40%",
      minWidth: "300px",
      marginBottom: "30px"
    },
    linkTag: {
      width: "16.6%",
      minWidth: "200px",
      paddingRight: "3px"
    }
  })
);

export default function Footer() {
  const classes = useStyles();
  
  return (
    <AppBar position="static" component="footer" className={classes.root}>
      <div className={classes.grid}>
        <div className={classes.siteTag}>
          <h2>Decision Trees</h2>
          <p>CSC 466 | Dr. Anderson | Spring 2021 | Cal Poly</p>
          <Typography variant="subtitle2">
            Ben Glossner | Ethan Zimbelman | Rupal Totale
          </Typography>
        </div>
        <div className={classes.linkTag}>
          <h3>Introduction</h3>
          <Link to='/getting-started'>Getting Started</Link>
          <Link to='/preliminary-skills'>Preliminary Skills</Link>
          <Link to='/introduction'>Intro to Decision Trees</Link>
        </div>
        
        <div className={classes.linkTag}>
          <h3>Tutorials</h3>
          {tutorials.map((tutorial) =>
            <Link to={`/${tutorial.notebook}`}>{tutorial.content.title}</Link>
          )}
        </div>
        
        <div className={classes.linkTag}>
          <h3>Resources</h3>
          <a
            href="https://github.com/CSC466-Team7/csc466_project"
            target="_blank"
            rel="noreferrer noopener"
          >
            GitHub
          </a>
          <a
            href="https://github.com/CSC466-Team7/csc466_project/tree/main/code/notebooks"
            target="_blank"
            rel="noreferrer noopener"
          >
            Notebooks
          </a>
          <a
            href="https://github.com/CSC466-Team7/csc466_project/tree/main/datasets"
            target="_blank"
            rel="noreferrer noopener"
          >
            Datasets
          </a>
        </div>
      
      </div>
    </AppBar>
  );
}
