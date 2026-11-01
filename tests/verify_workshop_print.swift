import AppKit
import Foundation
import PDFKit
import WebKit

// Run through verify_workshop_print.sh so the process also inherits a 190 MiB
// operating-system file-size limit. The verifier deletes its PDF on every exit.
let hardCapBytes = 199_229_440
let resultCapBytes = 10_485_760
let pageWidth: CGFloat = 720
let pageHeight: CGFloat = 960

final class PrintVerifier: NSObject, WKNavigationDelegate {
    let webView: WKWebView
    let outputURL: URL
    let outputDocument = PDFDocument()
    var captureBottom: CGFloat = 0
    var sourceDocumentHeight: CGFloat = 0
    var pageCount = 0
    var nextPage = 0
    var capturedBytes = 0
    var done = false
    var failure: String?
    var screenChecksStarted = false

    init(outputURL: URL) {
        self.outputURL = outputURL
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        webView = WKWebView(frame: NSRect(x: 0, y: 0, width: pageWidth, height: pageHeight), configuration: configuration)
        super.init()
        webView.navigationDelegate = self
    }

    func fail(_ message: String) {
        cleanupOutput()
        failure = message
        done = true
    }

    func cleanupOutput() {
        try? FileManager.default.removeItem(at: outputURL)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        fail(error.localizedDescription)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard !screenChecksStarted else { return }
        screenChecksStarted = true
        let screenChecks = #"""
        (() => {
          const supplement = document.getElementById('supplementary-material');
          const body = document.getElementById('supplement-body');
          const button = supplement?.querySelector(':scope > .fold-toggle');
          if (!supplement || !body || !button) return 'missing custom supplement fold';
          const changedBlocks = [...document.querySelectorAll('[data-review-changed-from="v1.3"]')];
          const highlights = [...document.querySelectorAll('.focus-highlight')];
          if (changedBlocks.length !== 66) return `unexpected changed-block count: ${changedBlocks.length}`;
          if (highlights.length !== 33) return `unexpected manual-focus count: ${highlights.length}`;
          if (document.querySelector('.revision-legend,.revision-swatch,.revision-light,.revision-medium,.revision-strong')) {
            return 'obsolete revision legend or importance styling remains';
          }
          if (changedBlocks.some(node => node.classList.contains('focus-highlight'))) {
            return 'a changed paragraph or block retains a whole-block yellow background';
          }
          if (highlights.some(node => ['transparent', 'rgba(0, 0, 0, 0)'].includes(getComputedStyle(node).backgroundColor))) {
            return 'word-level revision highlight is not visible on screen';
          }
          if (highlights.some(node => node.textContent.trim().length === 0 || node.textContent.trim().length > 180)) {
            return 'word-level highlight has an invalid text length';
          }
          if ([...document.querySelectorAll('.external-citation,.comment-anchor')].some(node => node.closest('.focus-highlight'))) {
            return 'citation link or comment marker is inside a yellow span';
          }
          const changedSample = ['P004', 'P010', 'P017', 'P025', 'P055', 'P066', 'P067', 'T2R1C2', 'supplement-data-access', 'P117', 'P119'];
          for (const id of changedSample) {
            const node = document.querySelector(`[data-review-block-id="${id}"]`);
            if (!node || node.getAttribute('data-review-changed-from') !== 'v1.3' || node.classList.contains('focus-highlight')) {
              return `changed sample metadata or word-level styling is incorrect: ${id}`;
            }
          }
          const unchangedSample = ['P002', 'P014', 'P016', 'T1R1C1'];
          for (const id of unchangedSample) {
            const node = document.querySelector(`[data-review-block-id="${id}"]`);
            if (!node || node.hasAttribute('data-review-changed-from') || node.hasAttribute('data-review-importance') || node.classList.contains('focus-highlight')) {
              return `unchanged sample is highlighted: ${id}`;
            }
          }
          const ablationPurpose = document.querySelector('[data-review-block-id="P067"]');
          if (!ablationPurpose?.textContent.includes('to examine how output quality changed across successive harness configurations')) {
            return 'revised ablation purpose is missing from the highlighted block';
          }
          if (document.body.innerText.includes('to examine how successive harness components contributed to the observed improvement')) {
            return 'obsolete ablation-purpose wording remains visible';
          }
          for (const id of ['P004', 'P008', 'P010', 'P012']) {
            const node = document.getElementById(id);
            const highlightedCharacters = [...node.querySelectorAll('.focus-highlight')].reduce((total, span) => total + span.textContent.trim().length, 0);
            if (!node || node.classList.contains('focus-highlight') || highlightedCharacters >= node.textContent.trim().length * 0.45) {
              return `Introduction paragraph does not use sparse manual focus: ${id}`;
            }
          }
          if (document.getElementById('P006')?.querySelector('.focus-highlight')) return 'simple P006 grammar edits are highlighted';
          const requiredFocusedText = [
            'focuses on that extraction task', 'outputs can vary',
            'checking every field in every output can approach the cost of manual extraction',
            'privacy and governance concerns', 'local models can still produce unstable or incorrect extractions',
            'without physician review', 'No physician reviewed pancreatic cancer outputs',
            'System identities', 'judge model and settings',
            'not provided to the models', 'not used during harness development',
            'not used to calculate the primary physician preference endpoint',
            'A displayed the inference harness output', 'B displayed the one prompt baseline',
            'blinded to system identities', 'Source attribution was available only with the harness output',
            'complete output packages'
          ];
          const highlightedText = new Set(highlights.map(node => node.textContent.trim()));
          if (requiredFocusedText.some(text => !highlightedText.has(text))) return 'a focused Methods or limitation phrase is missing';
          if (highlights.some(node => {
            const block = node.closest('[data-review-block-id]');
            const id = block?.dataset.reviewBlockId || '';
            return /^P\d{3}$/.test(id) && Number(id.slice(1)) >= 84 && Number(id.slice(1)) <= 119;
          })) return 'Results or Discussion contains a visible review focus';
          const comment13 = document.getElementById('comment-anchor-13');
          const comment14 = document.getElementById('comment-anchor-14');
          const comment13Target = 'We addressed this issue by incorporating treatment history and temporal context into the extraction process and adding verification steps to align treatment responses with the appropriate regimen.';
          if (!comment13?.previousSibling?.textContent?.endsWith(comment13Target)) return 'comment 13 anchor moved';
          if (!comment14?.previousSibling?.textContent?.includes('ER), progesterone receptor (PR), and HER2 status')) return 'comment 14 anchor moved';
          if (!document.getElementById('loo-results')?.querySelector('a[href="#supplementary-table-s3"]')) return 'independent LOO result is missing';
          for (const highlight of highlights) {
            const block = document.querySelector(`[data-review-block-id="${highlight.dataset.reviewFor}"]`);
            const sentences = block?.textContent.trim().split(/(?<=[.!?])\s+/).map(text => text.replace(/[.!?]+$/, '').trim()) || [];
            if (sentences.includes(highlight.textContent.trim().replace(/[.!?]+$/, '').trim())) return `complete sentence is highlighted: ${highlight.dataset.reviewFor}`;
          }
          for (const block of new Set(highlights.map(node => node.closest('[data-review-block-id]')))) {
            if (!block) return 'a focus span lacks its review block';
            const text = block.textContent;
            const ranges = [...block.querySelectorAll('.focus-highlight')].map(span => {
              const prefix = document.createRange();
              prefix.selectNodeContents(block);
              prefix.setEndBefore(span);
              const start = prefix.toString().length;
              return [start, start + span.textContent.length];
            }).sort((left, right) => left[0] - right[0]);
            const merged = [];
            for (const range of ranges) {
              const prior = merged[merged.length - 1];
              if (prior && !/[\p{L}\p{N}]/u.test(text.slice(prior[1], range[0]))) prior[1] = range[1];
              else merged.push([...range]);
            }
            const sentences = [];
            let sentenceStart = 0;
            for (const boundary of text.matchAll(/[.!?](?=\s|$)/g)) {
              let start = sentenceStart;
              let end = boundary.index;
              while (start < end && /\s/.test(text[start])) start += 1;
              while (end > start && /\s/.test(text[end - 1])) end -= 1;
              if (start < end) sentences.push([start, end]);
              sentenceStart = boundary.index + boundary[0].length;
            }
            let start = sentenceStart;
            let end = text.length;
            while (start < end && /\s/.test(text[start])) start += 1;
            while (end > start && /\s/.test(text[end - 1])) end -= 1;
            if (start < end) sentences.push([start, end]);
            if (merged.some(run => sentences.some(sentence => run[0] === sentence[0] && run[1] === sentence[1]))) {
              return `adjacent highlights visually cover a complete sentence: ${block.dataset.reviewBlockId}`;
            }
          }
          if (document.querySelector('.focus-highlight[data-review-for^="figure-"]')) return 'a figure caption is highlighted';
          const visible = () => getComputedStyle(body).display !== 'none';
          if (!supplement.classList.contains('is-collapsed') || visible() || button.getAttribute('aria-expanded') !== 'false') {
            return 'supplement was not initially collapsed';
          }
          button.click();
          if (!supplement.classList.contains('is-expanded') || !visible() || button.getAttribute('aria-expanded') !== 'true') {
            return 'supplement button did not expand';
          }
          button.click();
          const sectionLink = document.querySelector('.toc a[href="#supplementary-material"]');
          sectionLink?.click();
          if (!visible() || document.activeElement !== supplement) return 'supplement TOC link did not reveal and focus';
          button.click();
          const figureLink = document.querySelector('a[href="#figure-s1"]');
          figureLink?.click();
          if (!visible() || document.activeElement?.id !== 'figure-s1') return 'nested supplement link did not reveal and focus';
          button.click();
          const history = document.getElementById('version-history');
          const current = document.getElementById('version-1-6-3');
          const older = [...history.querySelectorAll('.version-entry')].filter(node => node !== current);
          if (history.open || !current?.open || older.some(node => node.open)) return 'unexpected initial version-history state';
          window.__workshopVersionHashState = 'waiting';
          window.addEventListener('hashchange', () => {
            setTimeout(() => {
              window.__workshopVersionHashState = 'settled';
            }, 0);
          }, {once: true});
          setTimeout(() => { location.hash='#version-1-4-1'; }, 0);
          return 'waiting-for-version-hash';
        })()
        """#
        webView.evaluateJavaScript(screenChecks) { value, error in
            if let error = error { self.fail(error.localizedDescription); return }
            guard (value as? String) == "waiting-for-version-hash" else {
                self.fail("Screen fold/navigation setup failed: \(value ?? "no result")"); return
            }
            self.waitForVersionHash(until: Date().addingTimeInterval(3))
        }
    }

    func waitForVersionHash(until deadline: Date) {
        webView.evaluateJavaScript("[window.__workshopVersionHashState || 'missing', location.hash, location.href].join(' | ')") { value, error in
            if let error = error { self.fail(error.localizedDescription); return }
            if (value as? String)?.hasPrefix("settled | ") == true {
                self.finishScreenChecks()
            } else if Date() >= deadline {
                self.fail("Version hashchange did not settle: \(value ?? "no result")")
            } else {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.02) {
                    self.waitForVersionHash(until: deadline)
                }
            }
        }
    }

    func finishScreenChecks() {
        let checks = #"""
        (() => {
          const history = document.getElementById('version-history');
          const current = document.getElementById('version-1-6-3');
          const prior = document.getElementById('version-1-4-1');
          const older = [...history.querySelectorAll('.version-entry')].filter(node => node !== current);
          if (location.hash !== '#version-1-4-1' || !history.open || current.open || !prior.open || older.some(node => node !== prior && node.open) || document.activeElement !== prior) {
            return 'version hash navigation did not isolate and focus prior entry';
          }
          for (const id of ['physician-comments', 'version-history', 'literature-workspace']) {
            const details = document.getElementById(id);
            const link = document.querySelector(`.toc a[href="#${id}"]`);
            if (!details || !link) return `missing screen fold ${id}`;
            details.open = false;
            link.click();
            if (!details.open || document.activeElement !== details) return `native fold link failed: ${id}`;
            details.open = false;
          }
          return 'ok';
        })()
        """#
        webView.evaluateJavaScript(checks) { value, error in
            if let error = error { self.fail(error.localizedDescription); return }
            guard (value as? String) == "ok" else {
                self.fail("Screen fold/navigation check failed: \(value ?? "no result")"); return
            }
            self.preparePrintExport()
        }
    }

    func preparePrintExport() {
        let printExportChecks = #"""
        (() => {
          const report = window.preparePrintExport(960);
          const highlights = [...document.querySelectorAll('.focus-highlight')];
          report.revisionHighlightPrintStyles = highlights.map(node => {
            const style = getComputedStyle(node);
            return {
              id: node.dataset.reviewFor || '',
              backgroundColor: style.backgroundColor,
              outlineStyle: style.outlineStyle,
              outlineWidth: style.outlineWidth,
              outlineColor: style.outlineColor,
            };
          });
          const printReviewRules = [];
          for (const sheet of document.styleSheets) {
            for (const rule of [...(sheet.cssRules || [])]) {
              if (!(rule instanceof CSSMediaRule) || rule.conditionText !== 'print') continue;
              for (const nested of [...rule.cssRules]) {
                if (nested.selectorText?.includes('.focus-highlight')) {
                  printReviewRules.push({
                    selector: nested.selectorText,
                    backgroundColor: nested.style.getPropertyValue('background-color'),
                    backgroundPriority: nested.style.getPropertyPriority('background-color'),
                    outline: nested.style.getPropertyValue('outline'),
                    outlinePriority: nested.style.getPropertyPriority('outline'),
                  });
                }
              }
            }
          }
          report.printReviewRules = printReviewRules;
          return report;
        })()
        """#
        webView.evaluateJavaScript(printExportChecks) { value, error in
            if let error = error { self.fail(error.localizedDescription); return }
            guard let report = value as? [String: Any],
                  let visible = report["supplementVisible"] as? Bool, visible,
                  let documentHeight = report["documentHeight"] as? Double,
                  let contentBottom = report["contentBottom"] as? Double,
                  let highlightCount = report["revisionHighlightCount"] as? Int,
                  let highlightBackgrounds = report["revisionHighlightBackgrounds"] as? [String],
                  let highlightPrintStyles = report["revisionHighlightPrintStyles"] as? [[String: Any]],
                  let printReviewRules = report["printReviewRules"] as? [[String: Any]],
                  let items = report["items"] as? [[String: Any]],
                  let offenders = report["offenders"] as? [String] else {
                self.fail("Invalid print-preparation report"); return
            }
            guard offenders.isEmpty else { self.fail("Page-boundary offenders: \(offenders)"); return }
            guard highlightCount == 33,
                  highlightBackgrounds.allSatisfy({ $0 == "transparent" || $0 == "rgba(0, 0, 0, 0)" }) else {
                self.fail("Revision highlights retained screen background in print export: \(highlightBackgrounds)"); return
            }
            guard highlightPrintStyles.count == 33,
                  highlightPrintStyles.allSatisfy({ style in
                      let background = style["backgroundColor"] as? String ?? ""
                      let outlineStyle = style["outlineStyle"] as? String ?? ""
                      let outlineWidth = style["outlineWidth"] as? String ?? ""
                      return (background == "transparent" || background == "rgba(0, 0, 0, 0)") &&
                             (outlineStyle == "none" || outlineWidth == "0px")
                  }) else {
                self.fail("Print export retained a highlight background or outline: \(highlightPrintStyles)"); return
            }
            guard printReviewRules.count == 1,
                  (printReviewRules[0]["backgroundColor"] as? String) == "transparent",
                  (printReviewRules[0]["backgroundPriority"] as? String) == "important",
                  ["0px", "none"].contains(printReviewRules[0]["outline"] as? String ?? ""),
                  (printReviewRules[0]["outlinePriority"] as? String) == "important" else {
                self.fail("Print media rule does not clear all highlight backgrounds and outlines: \(printReviewRules)"); return
            }
            let figureIDs = Set(items.filter { ($0["kind"] as? String) == "figure" }.compactMap { $0["id"] as? String })
            let tableIDs = Set(items.filter { ($0["kind"] as? String) == "table" }.compactMap { $0["id"] as? String })
            let expectedFigures: Set<String> = ["figure-1", "figure-2", "figure-3", "figure-4", "figure-s1", "figure-s2"]
            let expectedTables: Set<String> = ["table-1", "table-2", "table-3", "table-4", "table-5", "supplementary-table-s1", "supplementary-table-s2", "supplementary-table-s3"]
            guard figureIDs == expectedFigures else { self.fail("Unexpected print figures: \(figureIDs)"); return }
            guard tableIDs == expectedTables else { self.fail("Unexpected print tables: \(tableIDs)"); return }
            guard contentBottom > 0, contentBottom <= documentHeight + 1 else {
                self.fail("Invalid printable content boundary: \(contentBottom)/\(documentHeight)"); return
            }
            self.captureBottom = CGFloat(contentBottom)
            self.sourceDocumentHeight = CGFloat(documentHeight)
            self.pageCount = Int(ceil(self.captureBottom / pageHeight))
            guard (12...20).contains(self.pageCount) else { self.fail("Unexpected page count: \(self.pageCount)"); return }
            let finalPageHeight = self.captureBottom - CGFloat(self.pageCount - 1) * pageHeight
            guard finalPageHeight >= 80 else { self.fail("Trailing page is effectively empty: \(finalPageHeight) px"); return }
            self.captureNextPage()
        }
    }

    func captureNextPage() {
        if nextPage >= pageCount { finishPDF(); return }
        let y = CGFloat(nextPage) * pageHeight
        let height = min(pageHeight, captureBottom - y)
        let configuration = WKPDFConfiguration()
        configuration.rect = CGRect(x: 0, y: y, width: pageWidth, height: height)
        webView.createPDF(configuration: configuration) { result in
            switch result {
            case .failure(let error): self.fail(error.localizedDescription)
            case .success(let data):
                self.capturedBytes += data.count
                guard self.capturedBytes < hardCapBytes else { self.fail("Captured PDF data reached 190 MiB hard cap"); return }
                guard let pageDocument = PDFDocument(data: data), let page = pageDocument.page(at: 0) else {
                    self.fail("Could not decode captured page"); return
                }
                self.outputDocument.insert(page, at: self.outputDocument.pageCount)
                self.nextPage += 1
                self.captureNextPage()
            }
        }
    }

    func finishPDF() {
        guard let data = outputDocument.dataRepresentation() else { fail("Could not encode combined PDF"); return }
        guard data.count < resultCapBytes else { fail("Regression PDF exceeded 10 MiB"); return }
        let text = outputDocument.string ?? ""
        for required in ["Introduction", "Supplementary Methods", "Supplementary Table S1", "Supplementary Table S2", "Supplementary Table S3", "Supplementary Figure S1", "Supplementary Figure S2"] {
            guard text.contains(required) else { fail("Missing printed content: \(required)"); return }
        }
        for excluded in ["Author literature-review workspace", "Back to comment anchor", "Version history", "Draft relevance:"] {
            guard !text.contains(excluded) else { fail("Non-paper content was printed: \(excluded)"); return }
        }
        do {
            try data.write(to: outputURL, options: .atomic)
            if ProcessInfo.processInfo.environment["MED_DICT_PRINT_FAULT_AFTER_WRITE"] == "1" {
                fail("Injected post-write validation failure")
                return
            }
            let decoded = PDFDocument(url: outputURL)
            guard decoded?.pageCount == pageCount else { fail("Written PDF page count changed"); return }
            print("WK_PRINT_OK pages=\(pageCount) bytes=\(data.count) contentBottom=\(Int(captureBottom)) documentHeight=\(Int(sourceDocumentHeight)) figures=6 tables=8")
            done = true
        } catch {
            fail(error.localizedDescription)
        }
    }
}

let projectRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let inputURL = projectRoot.appendingPathComponent("WORKSHOP_PAPER_DRAFT.html")
guard FileManager.default.fileExists(atPath: inputURL.path) else {
    fputs("Run this verifier from the project root\n", stderr)
    exit(2)
}
let html: String
do {
    var selfContained = try String(contentsOf: inputURL, encoding: .utf8)
    let printFigurePaths = [
        "results/extraction_comparison/clinician_ratings/draft_figures/figure1_development_pathway_print.png",
        "results/extraction_comparison/clinician_ratings/draft_figures/figure2_evaluator_distribution_print.png",
        "results/extraction_comparison/clinician_ratings/draft_figures/figure3_interrater_matrix_print.png",
        "results/extraction_comparison/clinician_ratings/draft_figures/figure4_effect_profile_print.png",
        "results/extraction_comparison/clinician_ratings/draft_figures/supplementary_figure_s1_note_margins_print.png",
        "results/extraction_comparison/clinician_ratings/draft_figures/supplementary_figure_s2_staged_ablation_print.png",
    ]
    for relativePath in printFigurePaths {
        let figureURL = projectRoot.appendingPathComponent(relativePath)
        let figureData = try Data(contentsOf: figureURL)
        selfContained = selfContained.replacingOccurrences(
            of: "src=\"\(relativePath)\"",
            with: "src=\"data:image/png;base64,\(figureData.base64EncodedString())\""
        )
    }
    html = selfContained
} catch {
    fputs("Could not build self-contained WebKit input: \(error.localizedDescription)\n", stderr)
    exit(2)
}
let configuredOutputPath = ProcessInfo.processInfo.environment["MED_DICT_PRINT_OUTPUT_PATH"]
let outputURL = URL(fileURLWithPath: configuredOutputPath ?? "/private/tmp/med_dict_workshop_print_\(ProcessInfo.processInfo.processIdentifier).pdf")
try? FileManager.default.removeItem(at: outputURL)
let application = NSApplication.shared
application.setActivationPolicy(.prohibited)
let verifier = PrintVerifier(outputURL: outputURL)
verifier.webView.loadHTMLString(html, baseURL: projectRoot)
let deadline = Date().addingTimeInterval(180)
while !verifier.done && Date() < deadline {
    RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.05))
}
if !verifier.done { verifier.fail("WK print verification timed out") }
if let failure = verifier.failure {
    verifier.cleanupOutput()
    fputs("WK print verification failed: \(failure)\n", stderr)
    exit(failure == "WK print verification timed out" ? 3 : 1)
}
verifier.cleanupOutput()
