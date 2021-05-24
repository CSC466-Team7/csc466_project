import React from "react";
import { Card } from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    card: {
      width: "30%",
      minWidth: "300px",
      margin: "10px 8px",
    },
    "media": {
      height: "240px",
      width: "100%",
      display: "block",
      objectFit: "cover",
    },
    splash: {
      display: "block",
      margin: "40px auto",
    },
  }),
);


interface CardProps {
  children: React.ReactNode,
  img?: {
    url: string,
    title?: string,
  },
}

export default function CustomCard(props: CardProps) {
  const classes = useStyles();

  return (
    <Card className={classes.card}>
      {props.img && <img src={props.img.url} className={classes.media} />}
      {props.children}
    </Card>
  );
}
