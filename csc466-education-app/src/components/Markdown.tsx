import {
  Button,
  Checkbox,
  FormControlLabel,
  IconButton,
  Typography
} from "@material-ui/core";
import { FileCopy, GetApp } from "@material-ui/icons";
import AddIcon from "@material-ui/icons/Add";
import RemoveIcon from "@material-ui/icons/Remove";
import "katex/dist/katex.min.css";
import React, { useEffect, useState } from "react";
import { Collapse } from "react-bootstrap";
import { CopyToClipboard } from "react-copy-to-clipboard";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import dark from "react-syntax-highlighter/dist/esm/styles/prism/material-dark";
// @ts-ignore
import rehypeKatex from "rehype-katex";
// @ts-ignore
import rehypeRaw from "rehype-raw";
// @ts-ignore
import remarkMath from "remark-math";
import {
  DATASET_ASSETS_URL,
  MARKDOWN_ASSETS_URL,
  NOTEBOOK_ASSETS_URL
} from "../constants";
import Article from "./Article";
import Header from "./Header";
import "./Markdown.scss";
import Question, { TextOnlyQuestion, TextOnlyQuestionProps } from "./Question";
// import TEST_README from "../markdown/heart_classifier_with_sklearn.md";

function CodeComponent(props: any) {
  const [collapsed, setCollapsed] = useState(true);
  
  const changeCollapsed = () => {
    setCollapsed(!collapsed);
  };
  
  return (
    <div className="collapsable">
      <IconButton
        className="collapse-btn"
        onClick={changeCollapsed}>
        {!collapsed ?
          <AddIcon/> :
          <RemoveIcon/>
        }
      </IconButton>
      
      {!collapsed && <code className="hidden-text">Hidden</code>}
      
      <Collapse in={collapsed}>
        <div className="collapse-custom">
          <code
            className={props.className}
            children={props.children} {...props} />
        </div>
      </Collapse>
    </div>
  );
}

function DownloadButton(props: { href: string; children: React.ReactNode; }) {
  return (
    <Button
      component="a"
      color="primary"
      variant="contained"
      href={props.href}
      style={{margin: 10}}
      target={"_blank"}
      download
    >
      <GetApp/>
      {props.children}
    </Button>
  );
}

export interface InlineMarkdownQuestionProps {
  question: JSX.Element,
  answer: JSX.Element,
}

// @ts-ignore
const InlineMarkdownQuestion = (props: InlineMarkdownQuestionProps) => {
  return (
    <Question
      question={props.question}
      answer={props.answer}
      header={(
        <h3 className="mt-0"><u>Activity: Check Your Knowledge!</u></h3>
      )}
      styleOverrides={{
        width: "100%",
        marginTop: "5vh",
        marginBottom: "5vh",
      }}
    />
  );
};

export default function Markdown(props:
  {
    title: string,
    description: string,
    fileName: string;
    dataset?: string;
    questions: TextOnlyQuestionProps[]
  }) {
  const [markdownAsString, setMarkdownAsString] = useState("");
  const [isVersionClean, setVersionClean] = useState(true);
  
  const fetchMarkdown = async () => {
    const url = isVersionClean ? getCleanedFileURL() : getOriginalFileURL();
    // Switch when testing locally
    // const toFetch = TEST_README;
    const toFetch = url;
    const res = await fetch(toFetch);
    const resText = await res.text();
    setMarkdownAsString(resText);
  };
  
  const getPythonNotebookFileUrl = () => {
    return `${NOTEBOOK_ASSETS_URL}/${props.fileName}.ipynb`;
  };
  
  const getCleanedFileURL = () => {
    return `${MARKDOWN_ASSETS_URL}/${props.fileName}_cleaned.md`;
  };
  
  const getOriginalFileURL = () => {
    return `${MARKDOWN_ASSETS_URL}/${props.fileName}_original.md`;
  };
  
  useEffect(() => {
    fetchMarkdown();
  }, [isVersionClean, props.fileName]);
  
  const components: any = {
    // @ts-ignore
    code({node, inline, className, children, ...props}) {
      const match = /language-(\w+)/.exec(className || "");

      if (!inline && match) {
        return (
          <span>
            <CopyToClipboard text={String(children)}>
              <IconButton
                color="primary"
                component="span"
                style={{float: "right", marginLeft: 10}}>
                <FileCopy/>
              </IconButton>
            </CopyToClipboard>
            <SyntaxHighlighter
              style={dark} language={match[1]} PreTag="div"
              children={
                String(children)
                  .replace(/\n$/, "")
              }
              {...props} />
          </span>
        );
      }
      else if (!inline && !match) {
        return (
          <CodeComponent className={className} children={children} {...props} />
        );
      }
      
      return (
        <code
          className={className}
          children={children} {...props} />
      );
    },

    // @ts-ignore
    qinline({children}) {
      const filteredChildren = children.filter((c: any) => {
        return typeof(c) !== "string";
      });

      if (filteredChildren.length !== 2) {
        console.warn(children);
        throw new Error("Wrong number of children for qinline!");
      }

      return (
        <InlineMarkdownQuestion question={filteredChildren[0]} answer={filteredChildren[1]} />
      );
    },

    // @ts-ignore
    question({className, children, ...props}) {
      return (
        <div className={`${className} mr-1`} {...props}>
          {children}
        </div>
      );
    },

    // @ts-ignore
    answer({className, children, ...props}) {
      return (
        <p className={`${className}`} {...props}>
          {children}
        </p>
      );
    }
  };

  return (
    <>
      <Header description={props.description} title={props.title}/>
      <Article>
        <div style={{
          margin: 10,
          padding: 10,
          display: "flex",
          flexDirection: "row",
          justifyContent: "center"
        }}>
          <DownloadButton href={getPythonNotebookFileUrl()}>
            Download Python Notebook
          </DownloadButton>
          
          {props.dataset &&
          <DownloadButton href={`${DATASET_ASSETS_URL}/${props.dataset}`}>
            Download Dataset
          </DownloadButton>
          }
        </div>
        <FormControlLabel
          control={<Checkbox
            checked={!isVersionClean}
            onChange={() => setVersionClean(!isVersionClean)} name="checkedA"/>}
          label="Show all code"
        />
        <article className="markdown-body" style={{margin: 0, padding: 0}}>
          <ReactMarkdown
            components={components}
            skipHtml={false}
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeRaw, rehypeKatex]}>{markdownAsString}</ReactMarkdown>
        </article>
      </Article>
      <Typography variant={"h4"} style={{textAlign: "center"}}>
        Test your understanding
      </Typography>
      {props.questions.map((questionAndAnswer) =>
        <TextOnlyQuestion {...questionAndAnswer}/>)
      }
    </>
  );
}