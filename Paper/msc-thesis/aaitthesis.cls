%% aaitthesis.cls
%% MSc thesis class for the School of Information Technology and Engineering
%% (SiTE), Addis Ababa Institute of Technology (AAiT), Addis Ababa University.
%%
%% Formatting follows the SiTE "Guideline for MSc Thesis Proposal Writing and
%% Evaluation" (October 2022) where it applies to the thesis as well:
%%   * 12 pt type and 1.5 interline spacing
%%   * title page with AAU/AAiT logos, title, student, advisor(s), department,
%%     school, institute, university, month and year; no page number
%%   * section numbering 1. / 1.1.
%%   * IEEE referencing style
%% The thesis-only pages (approval by the board of examiners, declaration)
%% follow common AAU practice.
%%
%% Class options:
%%   guidance    (default) show the grey writing-guide boxes
%%   noguidance  hide them, which is what you submit
%%   Any other option is passed on to the standard report class.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{aaitthesis}[2026/11/06 v1.0 AAU AAiT SiTE MSc thesis]

\newif\ifaait@guidance
\aait@guidancetrue
\DeclareOption{guidance}{\aait@guidancetrue}
\DeclareOption{noguidance}{\aait@guidancefalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

\LoadClass[12pt,a4paper,oneside]{report}

%----------------------------------------------------------------------------
% Fonts, page layout and spacing
%----------------------------------------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage{amsmath}
\RequirePackage{newtxtext}   % Times-like text font
\RequirePackage{newtxmath}   % matching maths font (provides AMS symbols)
\RequirePackage{microtype}

% Wider left margin for binding. Adjust if your school asks for other values.
\RequirePackage[a4paper,top=2.5cm,bottom=2.5cm,left=3cm,right=2.5cm]{geometry}

% 1.5 interline spacing; setspace keeps floats and footnotes single spaced.
\RequirePackage{setspace}
\onehalfspacing

%----------------------------------------------------------------------------
% General packages
%----------------------------------------------------------------------------
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{array}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{multirow}
\RequirePackage{enumitem}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{csquotes}
\RequirePackage{fmtcount}
\RequirePackage[chapter]{algorithm} % numbered per chapter: Algorithm 4.1
\RequirePackage{algpseudocode}
\RequirePackage[breakable]{tcolorbox}
\RequirePackage[nohyperlinks]{acronym}

\graphicspath{{figures/}}
\setlist{itemsep=0pt,topsep=4pt}

% Captions: "Figure 1.1: ..." / "Table 1.1: ..."; table captions go on top.
\captionsetup{labelfont=bf,labelsep=colon,font=small}
\captionsetup[table]{position=top,skip=6pt}

%----------------------------------------------------------------------------
% References: IEEE style. Only works cited in the text are listed.
%----------------------------------------------------------------------------
\RequirePackage[backend=biber,style=ieee,dashed=false]{biblatex}

%----------------------------------------------------------------------------
% Headings
%   Chapters:     CHAPTER ONE / INTRODUCTION   (APPENDIX A / ... in appendices)
%   Sections:     1.1. Background
%   Unnumbered:   ABSTRACT, TABLE OF CONTENTS, REFERENCES, ...
%----------------------------------------------------------------------------
\RequirePackage{titlesec}
\setcounter{secnumdepth}{3}
\newcommand\aait@chapnum{\NUMBERstring{chapter}}
\titleformat{\chapter}[display]
  {\normalfont\bfseries\filcenter}
  {\large\MakeUppercase{\chaptertitlename}\ \aait@chapnum}
  {0.4\baselineskip}
  {\Large\MakeUppercase}
\titleformat{name=\chapter,numberless}[block]
  {\normalfont\Large\bfseries\filcenter}{}{0pt}{\MakeUppercase}
\titlespacing*{\chapter}{0pt}{0pt}{1.5\baselineskip}
\titleformat{\section}
  {\normalfont\large\bfseries}{\thesection.}{0.6em}{}
\titleformat{\subsection}
  {\normalfont\normalsize\bfseries}{\thesubsection.}{0.6em}{}
\titleformat{\subsubsection}
  {\normalfont\normalsize\bfseries\itshape}{\thesubsubsection.}{0.6em}{}
\titlespacing*{\section}{0pt}{1.2\baselineskip}{0.4\baselineskip}
\titlespacing*{\subsection}{0pt}{\baselineskip}{0.3\baselineskip}
\titlespacing*{\subsubsection}{0pt}{0.8\baselineskip}{0.2\baselineskip}

%----------------------------------------------------------------------------
% Table of contents and lists of figures / tables
%----------------------------------------------------------------------------
\RequirePackage[titles]{tocloft}
\setcounter{tocdepth}{2}   % chapters, sections and subsections
\renewcommand{\contentsname}{Table of Contents}
\renewcommand{\listfigurename}{List of Figures}
\renewcommand{\listtablename}{List of Tables}
\renewcommand{\cftchapaftersnum}{.}
\renewcommand{\cftsecaftersnum}{.}
\renewcommand{\cftsubsecaftersnum}{.}
\renewcommand{\cftchapleader}{\bfseries\cftdotfill{\cftdotsep}}
\setlength{\cftbeforechapskip}{4pt}
\setlength{\cftsecnumwidth}{2.6em}
\setlength{\cftsubsecnumwidth}{3.4em}
\renewcommand{\cftfigpresnum}{Figure~}
\renewcommand{\cftfigaftersnum}{:}
\setlength{\cftfignumwidth}{5.6em}
\setlength{\cftfigindent}{0pt}
\renewcommand{\cfttabpresnum}{Table~}
\renewcommand{\cfttabaftersnum}{:}
\setlength{\cfttabnumwidth}{5.2em}
\setlength{\cfttabindent}{0pt}

% Appendices: "APPENDIX A" headings, "Appendix A: ..." in the table of contents.
\g@addto@macro\appendix{%
  \gdef\aait@chapnum{\thechapter}%
  \addtocontents{toc}{%
    \protect\renewcommand{\protect\cftchappresnum}{\appendixname\space}%
    \protect\renewcommand{\protect\cftchapaftersnum}{:}%
    \protect\setlength{\protect\cftchapnumwidth}{6.6em}}}

%----------------------------------------------------------------------------
% Thesis information. Each field gets a user command (e.g. \advisor{...})
% that stores its value in \aait@<field>.
%----------------------------------------------------------------------------
\def\aait@field#1#2{%
  \@namedef{aait@#1}{#2}%
  \expandafter\def\csname #1\endcsname##1{\@namedef{aait@#1}{##1}}}

\aait@field{thesistitle}{[Title of the Thesis]}
\aait@field{studentname}{[Full Name of the Student]}
\aait@field{advisor}{[Full Name of the Advisor]}
\aait@field{coadvisor}{}
\aait@field{department}{[Name of the Department]}
\aait@field{degree}{Master of Science in [Program]}
\aait@field{school}{School of Information Technology and Engineering}
\aait@field{institute}{Addis Ababa Institute of Technology}
\aait@field{university}{Addis Ababa University}
\aait@field{place}{Addis Ababa, Ethiopia}
\aait@field{submissiondate}{\monthname~\the\year}
\aait@field{aaulogo}{figures/aau-logo.jpg}
\aait@field{aaitlogo}{figures/aait-logo.png}
\aait@field{internalexaminer}{}
\aait@field{externalexaminer}{}
\aait@field{chairperson}{}
\aait@field{schooldean}{}

\newcommand\monthname{\ifcase\month\or January\or February\or March\or
  April\or May\or June\or July\or August\or September\or October\or
  November\or December\fi}

% Use these in your own text if needed.
\newcommand{\thethesistitle}{\aait@thesistitle}
\newcommand{\thestudentname}{\aait@studentname}

%----------------------------------------------------------------------------
% Writing-guide boxes
%----------------------------------------------------------------------------
\newtcolorbox{aait@guidebox}{breakable,colback=black!4,colframe=black!30,
  boxrule=0.4pt,arc=1.5pt,left=5pt,right=5pt,top=3pt,bottom=3pt,
  before skip=6pt,after skip=8pt,
  fontupper=\small\setstretch{1.1}\color{black!80}}

% \guide{text}: what to write in this section. Hidden by the noguidance option.
\newcommand{\guide}[1]{%
  \ifaait@guidance\expandafter\@firstofone\else\expandafter\@gobble\fi
  {\begin{aait@guidebox}\textbf{Guide:}~#1\end{aait@guidebox}}}

%----------------------------------------------------------------------------
% Title page: first page, no page number.
%----------------------------------------------------------------------------
\newcommand\aait@logo[2]{%
  \IfFileExists{#1}%
    {\raisebox{-0.5\height}{\includegraphics[height=2.8cm]{#1}}}%
    {\fbox{\parbox[c][2.6cm][c]{2.8cm}{\centering\footnotesize\sffamily
      #2 logo\\[3pt]\scriptsize placeholder\\(see README)}}}}

\newcommand\aait@institutionblock{%
  {\large\bfseries
   \MakeUppercase{\aait@university}\par\vspace{3pt}
   \MakeUppercase{\aait@institute}\par\vspace{3pt}
   \MakeUppercase{\aait@school}\par}}

\newcommand{\maketitlepage}{%
  \pagenumbering{roman}%
  \thispagestyle{empty}%
  \begingroup
  \centering\setstretch{1.2}%
  \aait@logo{\aait@aaulogo}{AAU}\hspace{1.5cm}\aait@logo{\aait@aaitlogo}{AAiT}\par
  \vspace{0.8cm}
  \aait@institutionblock
  \vspace{3pt}
  {\large\bfseries \aait@department\par}
  \vfill
  {\LARGE\bfseries \aait@thesistitle\par}
  \vfill
  \parbox{0.85\textwidth}{\centering\setstretch{1.2}%
    A thesis submitted to the \aait@school, \aait@institute,
    \aait@university, in partial fulfillment of the requirements for the
    degree of \aait@degree}\par
  \vfill
  {\large By\par\vspace{2pt}\textbf{\aait@studentname}\par}
  \vfill
  {\large \textbf{Advisor:} \aait@advisor\par
   \ifx\aait@coadvisor\@empty\else
     \textbf{Co-advisor:} \aait@coadvisor\par\fi}
  \vfill
  {\large \aait@place\par\aait@submissiondate\par}
  \endgroup
  \clearpage}

%----------------------------------------------------------------------------
% Signature tables (approval and declaration pages)
%----------------------------------------------------------------------------
\newcommand\aait@sigline{\rule{0.92\linewidth}{0.4pt}}
\newcommand\aait@nameorline[1]{\ifx#1\@empty\aait@sigline\else#1\fi}
\newlength\aait@rowgap
\setlength\aait@rowgap{0.75cm}
% \aait@sigrow{role}{\name-macro}
\newcommand\aait@sigrow[2]{%
  #1 & \aait@nameorline{#2} & \aait@sigline & \aait@sigline \\[\aait@rowgap]}
% Same, but the row is left out when the name is empty (e.g. no co-advisor).
\newcommand\aait@optsigrow[2]{%
  \ifx#2\@empty\expandafter\@gobbletwo\else\expandafter\aait@sigrow\fi{#1}{#2}}
\newenvironment{aait@sigtable}
  {\par\noindent\setstretch{1.1}%
   \tabularx{\textwidth}{@{}>{\bfseries\raggedright}p{3.5cm}
                          >{\raggedright\arraybackslash}X p{3cm} p{2.2cm}@{}}
   & \textbf{Name} & \textbf{Signature} & \textbf{Date} \\[\aait@rowgap]}
  {\endtabularx\par}

%----------------------------------------------------------------------------
% Approval page: signed by the board of examiners. Names left empty in
% main.tex print as a blank line to be filled in by hand.
%----------------------------------------------------------------------------
\newcommand{\makeapprovalpage}{%
  \clearpage\phantomsection\addcontentsline{toc}{chapter}{Approval Page}%
  \begingroup
  \begin{center}\setstretch{1.2}%
    \aait@institutionblock
    \vspace{0.7cm}
    {\Large\bfseries \aait@thesistitle\par}
    \vspace{0.5cm}
    {\large By\par\textbf{\aait@studentname}\par}
    \vspace{0.6cm}
    {\large\bfseries Approval by Board of Examiners\par}
  \end{center}
  \vspace{0.2cm}
  \noindent This is to certify that the thesis prepared by
  \aait@studentname, entitled \enquote{\aait@thesistitle}, and submitted in
  partial fulfillment of the requirements for the degree of \aait@degree,
  complies with the regulations of the University and meets the accepted
  standards with respect to originality and quality.\par
  \vspace{0.6cm}
  \begin{aait@sigtable}
    \aait@sigrow{Advisor}{\aait@advisor}
    \aait@optsigrow{Co-advisor}{\aait@coadvisor}
    \aait@sigrow{Internal Examiner}{\aait@internalexaminer}
    \aait@sigrow{External Examiner}{\aait@externalexaminer}
    \aait@sigrow{Chairperson}{\aait@chairperson}
    \aait@sigrow{School Dean}{\aait@schooldean}
  \end{aait@sigtable}
  \endgroup
  \clearpage}

%----------------------------------------------------------------------------
% Declaration page: originality statement by the student and approval for
% submission by the advisor(s).
%----------------------------------------------------------------------------
\newcommand{\makedeclaration}{%
  \prelimchapter{Declaration}%
  \noindent I, the undersigned, declare that this thesis is my original work,
  that it has not been presented for a degree in this or any other
  university, and that all sources of material used for the thesis have
  been duly acknowledged.\par
  \vspace{0.4cm}
  \begin{aait@sigtable}
    \aait@sigrow{Student}{\aait@studentname}
  \end{aait@sigtable}
  \vspace{0.4cm}
  \noindent This thesis has been submitted for examination with my approval
  as university advisor.\par
  \vspace{0.4cm}
  \begin{aait@sigtable}
    \aait@sigrow{Advisor}{\aait@advisor}
    \aait@optsigrow{Co-advisor}{\aait@coadvisor}
  \end{aait@sigtable}
  \vspace{0.4cm}
  \noindent Place: \aait@place\par
  \noindent Date of submission: \aait@submissiondate\par
  \clearpage}

%----------------------------------------------------------------------------
% Preliminary pages (roman numbers) and main matter (arabic numbers)
%----------------------------------------------------------------------------
% \prelimchapter{Title}: unnumbered, centred heading that is listed in the
% table of contents (Abstract, Acknowledgment, List of Acronyms, ...).
\newcommand{\prelimchapter}[1]{%
  \clearpage\phantomsection\addcontentsline{toc}{chapter}{#1}\chapter*{#1}}

% List the lists of figures and tables in the table of contents.
\let\aait@listoffigures\listoffigures
\renewcommand{\listoffigures}{%
  \clearpage\phantomsection\addcontentsline{toc}{chapter}{\listfigurename}%
  \aait@listoffigures}
\let\aait@listoftables\listoftables
\renewcommand{\listoftables}{%
  \clearpage\phantomsection\addcontentsline{toc}{chapter}{\listtablename}%
  \aait@listoftables}

\newcommand{\mainmatter}{\clearpage\pagenumbering{arabic}}

% Unnumbered "REFERENCES" chapter, listed in the table of contents.
\newcommand{\printreferences}{%
  \printbibliography[heading=bibintoc,title={References}]}

%----------------------------------------------------------------------------
% Hyperlinks and cross-references (load last)
%----------------------------------------------------------------------------
\RequirePackage[hidelinks,bookmarksnumbered]{hyperref}
\RequirePackage[capitalise,noabbrev]{cleveref}
\crefname{algorithm}{Algorithm}{Algorithms}
\AtBeginDocument{%
  \hypersetup{pdftitle={\aait@thesistitle},pdfauthor={\aait@studentname}}}

\endinput
