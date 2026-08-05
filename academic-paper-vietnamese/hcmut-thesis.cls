\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hcmut-thesis}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{memoir}

\RequirePackage[utf8]{inputenc}
\RequirePackage{anyfontsize,mathptmx}
\RequirePackage[T1]{fontenc}

\RequirePackage{geometry,graphicx,indentfirst,setspace}
\RequirePackage{tikz}
\RequirePackage[center]{caption}

\RequirePackage{fancyhdr,hyperref}
\RequirePackage[super,square,sort]{natbib}

% Change to [english,vietnamese] to use Vietnamese labels
\RequirePackage[english,vietnamese]{babel}

% Custom commands
\newcommand{\Month}{
  \ifcase \the\month
  \or 01
  \or 02
  \or 03
  \or 04
  \or 05
  \or 06
  \or 07
  \or 08
  \or 09
  \or 10
  \or 11
  \or 12
  \fi
}

\newcommand{\@upperuniname}{Đại học Quốc gia Thành phố Hồ Chí Minh}
\newcommand{\@uniname}{Trường Đại học Bách Khoa}

\newcommand{\@deptname}{Khoa Khoa học \& Kỹ thuật Máy tính}
\newcommand{\deptname}[1]{\renewcommand{\@deptname}{#1}}

\newcommand{\@reporttype}{Luận văn tốt nghiệp}
\newcommand{\reporttype}[1]{\renewcommand{\@reporttype}{#1}}

\newcommand{\@stumajor}{Khoa học Máy tính}
\newcommand{\stumajor}[1]{\renewcommand{\@stumajor}{#1}}

\newcommand{\@majorid}{8480101}
\newcommand{\majorid}[1]{\renewcommand{\@majorid}{#1}}

\renewcommand{\@title}{Đề tài luận văn}

\newcommand{\@defcouncil}{Hội đồng: }
\newcommand{\defcouncil}[1]{\renewcommand{\@defcouncil}{#1}}

\newcommand{\@supervisor}{Giảng viên hướng dẫn: }
\newcommand{\supervisor}[1]{\renewcommand{\@supervisor}{#1}}

\newcommand{\@secretary}{Thư ký: }
\newcommand{\secretary}[1]{\renewcommand{\@secretary}{#1}}

\newcommand{\@reviewer}{Người phản biện: }
\newcommand{\reviewer}[1]{\renewcommand{\@reviewer}{#1}}

\newcommand{\@stuname}{Học viên: }
\newcommand{\stuname}[1]{\renewcommand{\@stuname}{#1}}

\newcommand{\@stuid}{Mã số học viên: }
\newcommand{\stuid}[1]{\renewcommand{\@stuid}{#1}}

\newcommand{\@reportplace}{Thành phố Hồ Chí Minh}
\newcommand{\reportplace}[1]{\renewcommand{\@reportplace}{#1}}

\newcommand{\@reporttime}{tháng 06 năm \the\year}
\newcommand{\reporttime}[1]{\renewcommand{\@reporttime}{#1}}

% Configurations
\usetikzlibrary{calc}

\geometry{
  paper=a4paper,
  left=3.5cm,
  right=2cm,
  vmargin=2cm,
  includeheadfoot=true,
  headheight=30pt
}

\renewcommand{\large}{\fontsize{15}{15}\selectfont}
\renewcommand{\Large}{\fontsize{20}{20}\selectfont}

\OnehalfSpacing
\raggedbottom
\renewcommand{\arraystretch}{1.2}

% Set depth of numbering for sections and table of contents
\setsecnumdepth{subsection}
\settocdepth{subsection}

% Set numwidths (see Section 9)
\setlength{\cfttablenumwidth}{3em}
\setlength{\cftfigurenumwidth}{3em}

\AtBeginDocument{\renewcommand*\bibname{References}}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=red,
  urlcolor=blue
}

\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}

\pagestyle{fancy}
\fancyhf{}

\if@twoside
  \fancyhead[EL,OR]{\bfseries\nouppercase\rightmark}
\else
  \fancyhead[R]{\bfseries\nouppercase\rightmark}
\fi

\fancyfoot[C]{\thepage}

% Custom pages and sections
\def\coverpage{
  \pagestyle{empty}
  \large\bfseries

  % \begin{tikzpicture}[remember picture,overlay]
  %   \draw[line width=3pt](
  %   $(current page.north west) + (3.0cm,-2.0cm)$
  %   ) rectangle (
  %   $(current page.south east) + (-2.0cm,2.0cm)$
  %   );
  %   \draw[line width=1pt](
  %   $(current page.north west) + (2.9cm,-1.9cm)$
  %   ) rectangle (
  %   $(current page.south east) + (-1.9cm,1.9cm)$
  %   );
  % \end{tikzpicture}

  \vspace*{-3\baselineskip}
  \begin{center}
    \MakeUppercase{\@upperuniname}

    \MakeUppercase{\@uniname}

    --------------------

    \vfill
    \MakeUppercase{\@stuname}
    \vfill
    {\Large\MakeUppercase{\parbox[c]{0.8\linewidth}{\centering\@title}}}
    \vfill
    {\normalsize
    \begin{tabular}{l}
        Chuyên ngành: {\@stumajor} \\
        Mã ngành: {\@majorid}
    \end{tabular}
    } \\[3cm]
    % {\mdseries Chuyên ngành: {\@stumajor}}

    % {\mdseries Mã ngành: {\@majorid}}
    \vfill
    \MakeUppercase{\@reporttype}
    \vfill
  \end{center}
  \vfill
  % \hfill\parbox[t]{13cm}{
  %   % THESIS COMMITTEE: \@defcouncil

  %   GVHD: \@supervisor
  %   % Giảng viên hướng dẫn: \parbox[t]{10cm}{\@supervisor}

  %   %THƯ KÝ: \@secretary
  %   % NGƯỜI PHẢN BIỆN: \@reviewer

  %   % Học viên: \@stuname

  %   Mã số học viên: \@stuid
  % }
  \vfill
  \mdseries
  \begin{center}
  \@reportplace, \@reporttime
  \end{center}
  \normalsize
  \cleardoublepage

  \pagestyle{fancy}
}

\providecommand{\phantomsection}{}
\newenvironment{preface}[1]{
  % \vspace*{\stretch{2}}
  \vspace{3cm}
  \thispagestyle{plain}
  {\noindent \bfseries \Huge #1}%
  \phantomsection\addcontentsline{toc}{chapter}{#1}
  \begin{center}
  \end{center}
}{
  \vspace*{\stretch{5}}
  \clearpage
}
