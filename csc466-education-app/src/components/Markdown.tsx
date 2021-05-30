import { IconButton, Paper, Button } from "@material-ui/core";
import { FileCopy } from "@material-ui/icons";
import AddIcon from "@material-ui/icons/Add";
import RemoveIcon from "@material-ui/icons/Remove";
import React, { useEffect, useState } from "react";
import { Collapse } from "react-bootstrap";
import { CopyToClipboard } from "react-copy-to-clipboard";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import dark from "react-syntax-highlighter/dist/esm/styles/prism/material-dark";
// @ts-ignore
import rehypeRaw from "rehype-raw";
import { DATASET_ASSETS_URL, MARKDOWN_ASSETS_URL } from "../constants";
// @ts-ignore
import remarkMath from "remark-math";
// @ts-ignore
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css"; // rehype-katex does not import the CSS for you
import "./Markdown.scss";

function CodeComponent(props: any) {
  const [collapsed, setCollapsed] = useState(true);
  
  const changeCollapsed = () => {
    setCollapsed(!collapsed);
  };
  
  return (
    <div className="collapsable">
      <IconButton className="collapse-btn"
        onClick={changeCollapsed}>
        {!collapsed ?
          <AddIcon/> :
          <RemoveIcon/>
        }
      </IconButton>

      {!collapsed && <code className="hidden-text">Hidden</code>}
      
      <Collapse in={collapsed}>
        <div className="collapse-custom">
          <code className={props.className}
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
      download
    >
      {props.children}
    </Button>
  );
}

export default function Markdown(props: { fileName: string; version?: string; dataset?: string; }) {
  const [markdownAsString, setMarkdownAsString] = useState("");

  const fetchMarkdown = async () => {
    const url = getExpectedFileURL();
    // Switch when testing locally
    // const file = await
    // import("../markdown/heart_decision_tree_classifier.md"); const toFetch =
    // file.default;
    const toFetch = url;
    const res = await fetch(toFetch);
    const resText = await res.text();
    setMarkdownAsString(resText);
  };

  const getExpectedFileURL = () => {
    if (props?.version === "original") {
      return getOriginalFileURL();
    }

    return getCleanedFileURL();
  };

  const getCleanedFileURL = () => {
    return `${MARKDOWN_ASSETS_URL}/${props.fileName}_cleaned.md`;
  };

  const getOriginalFileURL = () => {
    return `${MARKDOWN_ASSETS_URL}/${props.fileName}_original.md`;
  };

  useEffect(() => {
    fetchMarkdown();
  }, []);

  const components: any = {
    // @ts-ignore
    code({node, inline, className, children, ...props}) {
      const match = /language-(\w+)/.exec(className || "");

      if (!inline && match) {
        return (
          <span>
            <CopyToClipboard text={String(children)}>
              <IconButton color="primary"
                component="span"
                style={{float: "right", marginLeft: 10}}>
                <FileCopy/>
              </IconButton>
            </CopyToClipboard>
            <SyntaxHighlighter style={dark} language={match[1]} PreTag="div"
              children={
                String(children)
                  .replace(/\n$/, "")
              }
              {...props} />
          </span>
        );
      } else if (!inline && !match) {
        return (
          <CodeComponent className={className} children={children} {...props} />
        );
      }

      return (
        <code className={className}
          children={children} {...props} />
      );
    }
  };
  return (
    <Paper elevation={3} style={{padding: 20, margin: 20}}>
      <section className="file-options">
        <div>
          <DownloadButton href={getExpectedFileURL()}>
            Download
          </DownloadButton>

          {props?.version === "original" ? (
            <DownloadButton href={getCleanedFileURL()}>
              Download Cleaned
            </DownloadButton>
          ) : (
            <DownloadButton href={getOriginalFileURL()}>
              Download Full
            </DownloadButton>
          )}

          {props?.dataset &&
            <DownloadButton href={`${DATASET_ASSETS_URL}/${props.dataset}`}>
              Download Dataset
            </DownloadButton>
          }
        </div>
      </section>
      <article className="markdown-body">
        <ReactMarkdown
          components={components}
          skipHtml={false}
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeRaw, rehypeKatex]}>{markdownAsString}</ReactMarkdown>
      </article>
    </Paper>
  );
}