import React from "react";
import { Link } from "react-router-dom";
import {
  Button,
  CardActions,
  CardContent,
  CardMedia,
} from "@material-ui/core";

import Gallery from "../../components/Gallery";
import Card from "../../components/Card";

export default function ExampleList() {
  const ex1 = {
    url: "https://cdn.xxl.thumbs.canstockphoto.com/number-1-sign-design-template-element-black-icon-on-transparent-background-vector-clip-art_csp44589215.jpg",
    title: "the number 1",
  };

  const ex2 = {
    url: "https://snapchatemojis.com/wp-content/uploads/2015/09/2-fingers.png",
    title: "the number 2",
  };

  return (
    <>
      <section>
        <h2>Examples</h2>
        <p>
          Get into the learn by doing spirit with these walkthrough examples.
        </p>
      </section>

      <Gallery>
        <Card img={ex1}>
          <CardContent>
            <h3>Example #1</h3>
            <p>Info about the first example</p>
          </CardContent>
          <CardActions>
            <Button
              color="primary"
              variant="contained"
              href="#/example/1"
            >
              Example Info
            </Button>
          </CardActions>
        </Card>

        <Card img={ex2}>
          <CardContent>
            <h3>Example #2</h3>
            <p>Check out this example</p>
          </CardContent>
          <CardActions>
            <Button
              color="primary"
              variant="contained"
              href="#/example/2"
            >
              Example Info
            </Button>
          </CardActions>
        </Card>
      </Gallery>

      <article>
        <h2>Resources</h2>
        <p>
          Check the following out if you're having trouble working through the
          examples:
        </p>

        <ul>
          <li>
            <Link to="/introduction">Introduction to decision trees</Link>
          </li>
          <li>
            <Link to="/preliminary-skills">Preliminary skills refresher</Link>
          </li>
        </ul>
      </article>
    </>
  );
}
