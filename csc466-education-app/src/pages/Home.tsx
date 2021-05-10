import React from "react";
import {
  Button,
  Card,
  CardActions,
  CardContent,
  CardMedia,
} from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";

import CTACard from "../components/CTACard";
import Gallery from "../components/Gallery";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    card: {
      width: "30%",
    },
    media: {
      height: "240px",
    },
    splash: {
      display: "block",
      margin: "40px auto",
    },
  }),
);

export default function Home() {
  const classes = useStyles();

  return (
    <>
      <section>
        <h1>Your Intro to Decesion Trees</h1>
        <p>An explainable, data-driven approach to making decesions</p>
        <Button
          component="button"
          color="primary"
          variant="contained"
          href="/#/getting-started"
        >
          Get Started
        </Button>
      </section>

      <img
        className={classes.splash}
        src="https://www.decadeonrestoration.org/themes/unrestore/images/tree.jpg"
        alt="tree"
        width="800"
      />

      <CTACard>
        <span>
          <h2>New to Decision Trees?</h2>
          <p>Learn about them here</p>
        </span>
        <Button
          component="button"
          color="primary"
          variant="contained"
          href="/#/introduction"
        >
          Introduction
        </Button>
      </CTACard>

      <section>
        <h2>Want Practice?</h2>
        <p>It’s time to get your hands dirty with some interactive examples to really solidify your learning</p>
      </section>

      <Gallery>
        <Card className={classes.card}>
          <CardMedia
            className={classes.media}
            image="https://placekitten.com/640/360"
            title="placeholder image"
          />
          <CardContent>
            <h3>Getting Started</h3>
            <p>Everything you need to know to set up your environment</p>
          </CardContent>
        </Card>

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
        </Card>
      </Gallery>

      <CTACard>
        <span>
          <h2>Need a refresher?</h2>
          <p>Brush up on your preliminary skills</p>
        </span>
        <Button
          component="button"
          color="primary"
          variant="contained"
          href="/#/preliminary-skills"
        >
          Preliminary Skills
        </Button>
      </CTACard>
    </>
  );
}
