import { Paper } from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import React from "react";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    article: {
      margin: "25px auto",
      padding: "25px",
      width: "80%",
      backgroundColor: "whitesmoke"
    }
  })
);

interface ArticleProps {
  children: React.ReactNode,
}

export default function Article(props: ArticleProps) {
  const classes = useStyles();
  
  return (
    <Paper elevation={3} className={classes.article}>
      {props.children}
    </Paper>
  );
}
