import { Typography } from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import React from "react";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({})
);

interface HeaderProps {
  description: string;
  title: string;
  
}

export default function Header(props: HeaderProps) {
  const classes = useStyles();
  
  return (
    <>
      <section style={{width: "50%", margin: "20px auto"}}>
        <h1>{props.title}</h1>
        <Typography variant="subtitle1">{props.description}</Typography>
        <hr/>
      </section>
    </>
  );
}
