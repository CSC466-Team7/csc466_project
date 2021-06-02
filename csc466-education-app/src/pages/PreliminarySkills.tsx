import { Typography } from "@material-ui/core";
import React from "react";

import Gallery from "../components/Gallery";
import Header from "../components/Header";
import { skills } from "../data";

export default function PreliminarySkills() {
  
  return (
    <>
      <Header
        title={"Preliminary Skills"}
        description={"Need a quick brush up on some of the KDD tools used to build" +
        " decesion trees? Look no further! \n" +
        "Here are links to tutorials of some commonly used tools in data" +
        " science. \n"}/>
      
      <Gallery cards={skills}/>
      
      <section>
        <p>
          While these won't cover everything you'll need to know, hopefully
          they can serve as a nice refresher.
        </p>
      </section>
    </>
  );
}
