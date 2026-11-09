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
          {
          const expectedUnderlines = [
            ['author-review-1', 'a broader multistep workflow developed from physician feedback can be used to improve note extraction'],
            ['author-review-2a', 'one prompt baseline'],
            ['author-review-2b', 'single-prompt baseline'],
            ['author-review-2c', 'one prompt baseline'],
            ['author-review-3', 'population-averaged logistic generalized estimating equation (GEE) with an exchangeable working correlation structure at the note level'],
            ['author-review-4', 'to account for clustering of repeated field-level judgments within the same note'],
            ['author-review-5a', 'p < 0.05'],
            ['author-review-5b', 'p < 0.05'],
            ['author-review-5c', 'p < 0.05'],
            ['author-review-6', 'P < 0.05'],
            ['author-review-7', 'significantly affected the LLM judge’s preference'],
            ['author-review-2d', 'one prompt baseline'],
            ['author-review-8', 'improve the clinical quality of extracted information'],
            ['author-review-9', 'LLM-as-a-judge'],
            ['author-review-10', 'the clinical rules developed from breast cancer did not provide further improvement for pancreatic cancer'],
            ['author-review-11', 'some rules developed for breast cancer were specific to estrogen receptor (ER), progesterone receptor (PR), and HER2 status'],
          ];
          const underlines = [...document.querySelectorAll('.author-review-underline')];
          if (document.querySelector('.focus-highlight,.revision-legend,.revision-swatch,.revision-light,.revision-medium,.revision-strong')) {
            return 'legacy yellow review styling remains';
          }
          if (underlines.length !== expectedUnderlines.length) return `unexpected author underline count: ${underlines.length}`;
          for (let index = 0; index < expectedUnderlines.length; index += 1) {
            const [id, text] = expectedUnderlines[index];
            const node = underlines[index];
            if (node.id !== id || node.textContent.trim() !== text) return `author underline differs at ${id}`;
            if (!getComputedStyle(node).textDecorationLine.includes('underline')) return `author underline is not visible: ${id}`;
            if (node.textContent.trim().length === 0 || node.textContent.trim().length > 180) return `author underline has invalid length: ${id}`;
          }
          const reviewList = document.getElementById('author-review-list');
          if (!reviewList || reviewList.open) return 'author review list is not initially collapsed';
          const reviewSummary = reviewList.querySelector(':scope > summary');
          reviewSummary?.click();
          if (!reviewList.open) return 'author review list did not open manually';
          for (const [id] of expectedUnderlines) {
            if (!reviewList.querySelector(`a[href="#${id}"]`)) return `author review link is missing: ${id}`;
            if (!document.getElementById(id)?.parentElement?.querySelector('a[href="#author-review-list"]')) return `author review backlink is missing: ${id}`;
          }
          const discussion = document.getElementById('section-discussion');
          const reviewTarget = document.getElementById('author-review-10');
          discussion.open = false;
          reviewList.querySelector('a[href="#author-review-10"]')?.click();
          if (!discussion.open || document.activeElement !== reviewTarget) return 'author review link did not reveal its manuscript section';
          reviewSummary?.click();
          if (reviewList.open) return 'author review list did not close manually';
          discussion.open = false;
          window.history.replaceState(null, '', '#author-review-10');
          window.dispatchEvent(new HashChangeEvent('hashchange'));
          if (!discussion.open || document.activeElement !== reviewTarget) return 'author review URL did not reveal its manuscript section';
          if (document.querySelectorAll('.figure').length !== 6) return 'unexpected manuscript figure count';
          if (document.querySelectorAll('.table-wrap').length !== 8) return 'unexpected manuscript table count';
          if (document.querySelectorAll('#physician-comments .comment').length !== 0) return 'source comments remain visible';
          if (document.body.innerText.includes('Difference from 50%')) return 'deleted difference-from-50 wording remains';
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
          const current = document.getElementById('version-2-0');
          const older = [...history.querySelectorAll('.version-entry')].filter(node => node !== current);
          if (history.open || !current?.open || older.some(node => node.open)) return 'unexpected initial version-history state';
          window.__workshopVersionHashState = 'waiting';
          window.addEventListener('hashchange', () => {
            setTimeout(() => { window.__workshopVersionHashState = 'settled'; }, 0);
          }, {once: true});
          setTimeout(() => { location.hash='#version-1-8-1'; }, 0);
          return 'waiting-for-version-hash';
          }
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
          const current = document.getElementById('version-2-0');
          const prior = document.getElementById('version-1-8-1');
          const older = [...history.querySelectorAll('.version-entry')].filter(node => node !== current);
          if (location.hash !== '#version-1-8-1' || !history.open || current.open || !prior.open || older.some(node => node !== prior && node.open) || document.activeElement !== prior) {
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
          report.authorUnderlineCount = document.querySelectorAll('.author-review-underline').length;
          const p114 = document.getElementById('P114');
          report.v2PrintContent = Boolean(
            p114 && getComputedStyle(p114).display !== 'none' &&
            p114.textContent.includes('four physicians') &&
            p114.textContent.includes('challenging')
          );
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
                  let authorUnderlineCount = report["authorUnderlineCount"] as? Int,
                  let v2PrintContent = report["v2PrintContent"] as? Bool,
                  let items = report["items"] as? [[String: Any]],
                  let offenders = report["offenders"] as? [String] else {
                self.fail("Invalid print-preparation report"); return
            }
            guard offenders.isEmpty else { self.fail("Page-boundary offenders: \(offenders)"); return }
            guard v2PrintContent else { self.fail("v2.0 Discussion text is missing from print export"); return }
            guard authorUnderlineCount == 16 else { self.fail("Author review underline count changed in print export: \(authorUnderlineCount)"); return }
            guard highlightCount == 0, highlightBackgrounds.isEmpty, highlightPrintStyles.isEmpty else {
                self.fail("Legacy yellow revision marks remain in print export"); return
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
        for required in ["Introduction", "This study has several limitations", "Supplementary Methods", "Supplementary Table S1", "Supplementary Table S2", "Supplementary Table S3", "Supplementary Figure S1", "Supplementary Figure S2"] {
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
