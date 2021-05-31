import {
  Button,
  Card,
  CardActions,
  CardContent,
  CardMedia
} from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import { KeyboardArrowRightOutlined } from "@material-ui/icons";
import React from "react";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    card: {
      width: "30%",
      minWidth: "300px",
      margin: "15px"
    },
    "media": {
      height: "240px",
      width: "100%",
      display: "block",
      objectFit: "cover",
      border: "3px solid",
      borderColor: "#4a4848"
    },
    splash: {
      display: "block",
      margin: "40px auto"
    }
  })
);

interface CardProps {
  img: {
    url: string,
    title: string,
  },
  content: {
    title: string,
    description: string
  },
  linkTo: string
}

export default function CustomCard(props: CardProps) {
  const classes = useStyles();
  
  return (
    <Card className={classes.card}>
      <CardMedia
        className={classes.media}
        image={props.img.url}
        title={props.img.title}
      />
      <CardContent>
        <h3>{props.content.title}</h3>
        <p>{props.content.description}</p>
      </CardContent>
      <CardActions>
        <Button
          color="primary"
          variant="contained"
          href={props.linkTo}
        >
          View Tutorial
          <KeyboardArrowRightOutlined/>
        </Button>
      </CardActions>
    </Card>
  );
}
