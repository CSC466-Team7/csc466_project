import React from "react";

import Gallery from "../components/Gallery";
import { skills } from "../data";

export default function PreliminarySkills() {
  
  return (
    <>
      <section>
        <h2>Preliminary Skills</h2>
        <p>
          Need a quick brush up on some of the KDD tools used to build decesion
          trees? Look no further!<br/>
          Here are links to tutorials of some commonly used tools in data
          science.
        </p>
      </section>
      
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
