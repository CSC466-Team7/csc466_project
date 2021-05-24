import React from "react";

import {
  Button,
  CardActions,
  CardContent,
  CardMedia,
} from "@material-ui/core";

import Gallery from "../components/Gallery";
import Card from "../components/Card";

export default function PreliminarySkills() {
  const rubix = {
    url: "https://www.megahowto.com/wp-content/uploads/2009/09/Rubix-Cube.jpg",
    title: "a rubix cube",
  };

  const pandas = {
    url: "https://cameoglassuk.co.uk/wp-content/uploads/2016/07/EATING-PANDAS-1.jpg",
    title: "pandas eating bamboo",
  };

  const clustering = {
    url:"https://miro.medium.com/max/1000/1*0DDt5Xp9z6ecj5eL6FNAfQ.png",
    title:"data clustering example",
  };

  return (
    <>
      <section>
        <h2>Preliminary Skills</h2>
        <p>
          Need a quick brush up on some of the KDD tools used to build decesion
          trees? Look no further!<br />
          Here are links to tutorials of some commonly used tools in data science.
        </p>
      </section>

      <Gallery>
        <Card img={rubix}>
          <CardContent>
            <h3>Numpy</h3>
            <p>A library for easily handling multi-dimensional arrays and
            matrices.</p>
          </CardContent>
          <CardActions>
            <Button
              color="primary"
              variant="contained"
              href="https://numpy.org/doc/stable/user/absolute_beginners.html"
              target="_blank"
            >
              View Tutorial
            </Button>
          </CardActions>
        </Card>

        <Card img={pandas}>
          <CardContent>
            <h3>Pandas</h3>
            <p>A library for powerful data analysis and manipulation.</p>
          </CardContent>
          <CardActions>
            <Button
              color="primary"
              variant="contained"
              href="https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html"
              target="_blank"
            >
              View Tutorial
            </Button>
          </CardActions>
        </Card>

        <Card img={clustering}>
          <CardContent>
            <h3>Scikit Learn</h3>
            <p>A library that makes ML processes (like dimensionality reduction)
            easy.</p>
          </CardContent>
          <CardActions>
            <Button
              color="primary"
              variant="contained"
              href="https://scikit-learn.org/stable/tutorial/index.html"
              target="_blank"
            >
              View Tutorial
            </Button>
          </CardActions>
        </Card>
      </Gallery>

      <section>
        <p>
          While these won't cover everything you'll need to know, hopefully
          they can serve as a nice refresher.
        </p>
      </section>
    </>
  );
}
