import React from "react";
import { Switch, Route, HashRouter } from "react-router-dom";
import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";
import { Container } from "@material-ui/core/";
import Markdown from "./components/Markdown";

import Home from "./pages/Home";
import Introduction from "./pages/Introduction";
import GettingStarted from "./pages/GettingStarted";
import PreliminarySkills from "./pages/PreliminarySkills";

import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import "./App.css";
import "bootstrap/dist/css/bootstrap.min.css";

const theme = createMuiTheme({
  palette: {
    primary: {
      light: "#62727b",
      main: "#37474f",
      dark: "#102027",
      contrastText: "#fff",
    },
    secondary: {
      light: "#60ad5e",
      main: "#2e7d32",
      dark: "#005005",
      contrastText: "#000",
    },
  },
});

function App() {
  return (
    <HashRouter>
      <ThemeProvider theme={theme}>
        <Navbar />
        <Container>
          <Switch>
            <Route path="/example-markdown">
              <Markdown fileName={"running-code"}/>
            </Route>
            <Route exact path="/" component={Home} />
            <Route path="/introduction" component={Introduction} />
            <Route path="/getting-started" component={GettingStarted} />
            <Route path="/preliminary-skills" component={PreliminarySkills} />
            <Route path="/heart-disease">
              <Markdown fileName={"heart_decision_tree_classifier"} />
            </Route>
          </Switch>
        </Container>
        <Footer />
      </ThemeProvider>
    </HashRouter>
  );
}

export default App;
