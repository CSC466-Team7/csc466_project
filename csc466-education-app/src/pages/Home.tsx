import {
  Button,
  Card,
  CardActions,
  CardContent,
  CardMedia
} from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import React from "react";

import CTACard from "../components/CTACard";
import Gallery from "../components/Gallery";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    card: {
      width: "30%",
      minWidth: "300px",
      margin: "10px 8px"
    },
    media: {
      height: "240px"
    },
    splash: {
      display: "block",
      margin: "40px auto",
      width: "80%",
      backgroundColor: "#cccccc"
    },
    backgroundImage: {
      position: "fixed",
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      opacity: 0.25,
      zIndex: -100,
      height: "100%",
      background: "url(https://www.decadeonrestoration.org/themes/unrestore/images/tree.jpg) no-repeat center center fixed",
      backgroundSize: "cover"
    },
    overlay: {
      position: "fixed",
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      opacity: 0.35,
      zIndex: -100,
      height: "100%",
      backgroundColor: "#000000",
      backgroundSize: "cover"
    }
  })
);

export default function Home() {
  const classes = useStyles();
  
  return (
    <>
      <div className={classes.backgroundImage}/>
      <div className={classes.overlay}/>
      <section>
        <h1>Your Intro to Decision Trees</h1>
        <p>An explainable, data-driven approach to making decisions</p>
      </section>
      <CTACard
        title="Not sure where to start?"
        description="Learn how to use the website"
        buttonText="Get Started"
        linkTo="/#/getting-started"/>
      <CTACard
        title="New to Decision Trees?"
        description="Learn the theory behind decision trees"
        buttonText="Introduction"
        linkTo="/#/introduction"/>
      <CTACard
        title="Need a refresher?"
        description="Brush up on your preliminary skills"
        buttonText="Preliminary Skills"
        linkTo="/#/preliminary-skills"/>
      
      <section>
        <h2>Want to put it into practice?</h2>
        <p>Follow the guides below to implement and test powerful machine learning models to
          better understand decision trees</p>
      </section>
      <Gallery>
        <Card className={classes.card}>
          <CardMedia
            className={classes.media}
            image="https://placekitten.com/640/360"
            title="placeholder image"
          />
          <CardContent>
            <h3>[Example 1]</h3>
            <p>Description of [Example 1]</p>
          </CardContent>
          <CardActions>
            <Button
              color="primary"
              variant="contained"
              href="/#/example/1"
            >
              See Example #1
            </Button>
          </CardActions>
        </Card>
        
        <Card className={classes.card}>
          <CardMedia
            className={classes.media}
            image="https://placekitten.com/640/360"
            title="placeholder image"
          />
          <CardContent>
            <h3>[Example 2]</h3>
            <p>Description of [Example 2]</p>
          </CardContent>
          <CardActions>
            <Button
              color="primary"
              variant="contained"
              href="/#/example/2"
            >
              See Example #2
            </Button>
          </CardActions>
        </Card>
      </Gallery>
    </>
  );
}
