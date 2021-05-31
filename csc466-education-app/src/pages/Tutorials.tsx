import React from "react";
import { Link } from "react-router-dom";
import {
  Button,
  CardActions,
  CardContent,
  CardMedia,
} from "@material-ui/core";

import Gallery from "../components/Gallery";
import { skills, tutorials } from "../data";

export default function Tutorials() {
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
        <h2>Tutorials</h2>
        <p>
          Get into the learn by doing spirit with these walkthrough tutorials.
        </p>
      </section>
  
      <Gallery cards={tutorials}/>

      <article>
        <h2>Resources</h2>
        <p>
          Check the following out if you're having trouble working through the
          tutorials:
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
