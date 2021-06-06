import {
  Button,
  Card,
  CardActions,
  CardContent,
  CardMedia
} from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import { KeyboardArrowRightOutlined, OpenInNew } from "@material-ui/icons";
import React from "react";
import { Link } from "react-router-dom";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    card: {
      width: "30%",
      minWidth: "500px",
      margin: "15px",
      display: "flex",
      flexDirection: "column"
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

export interface CardProps {
  img: {
    url: string,
    title: string,
  },
  content: {
    title: string,
    description: string
  },
  linkTo?: string,
  notebook?: string
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
      <CardActions style={{
        display: "flex",
        justifyContent: "flex-end",
        alignItems: "flex-end",
        flex: "1 1 auto",
      }}>
        {/*If external link...*/}
        {props.linkTo &&
        <Button
          color="primary"
          variant="contained"
          href={props.linkTo}
          style={{margin: 10}}
          target="_blank">
          View Tutorial
          <OpenInNew/>
        </Button>}
        {/*If link to notebook...*/}
        {props.notebook && <Button
          component={Link}
          color="primary"
          variant="contained"
          to={`/${props.notebook}`}
          style={{margin: 10}}>
          View Tutorial
          <KeyboardArrowRightOutlined/>
        </Button>}
      
      </CardActions>
    </Card>
  );
}
