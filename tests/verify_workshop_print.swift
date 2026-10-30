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
        let screenChecks = #"""
        (() => {
          const supplement = document.getElementById('supplementary-material');
          const body = document.getElementById('supplement-body');
          const button = supplement?.querySelector(':scope > .fold-toggle');
          if (!supplement || !body || !button) return 'missing custom supplement fold';
          const highlights = [...document.querySelectorAll('mark.revision-highlight')];
          if (highlights.length !== 3) return `unexpected revision-highlight count: ${highlights.length}`;
          if (highlights.some(node => ['transparent', 'rgba(0, 0, 0, 0)'].includes(getComputedStyle(node).backgroundColor))) {
            return 'revision highlight is not visible on screen';
          }
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
        webView.evaluateJavaScript(screenChecks) { value, error in
            if let error = error { self.fail(error.localizedDescription); return }
            guard (value as? String) == "ok" else {
                self.fail("Screen fold/navigation check failed: \(value ?? "no result")"); return
            }
            self.preparePrintExport()
        }
    }

    func preparePrintExport() {
        webView.evaluateJavaScript("window.preparePrintExport(960)") { value, error in
            if let error = error { self.fail(error.localizedDescription); return }
            guard let report = value as? [String: Any],
                  let visible = report["supplementVisible"] as? Bool, visible,
                  let documentHeight = report["documentHeight"] as? Double,
                  let contentBottom = report["contentBottom"] as? Double,
                  let highlightCount = report["revisionHighlightCount"] as? Int,
                  let highlightBackgrounds = report["revisionHighlightBackgrounds"] as? [String],
                  let items = report["items"] as? [[String: Any]],
                  let offenders = report["offenders"] as? [String] else {
                self.fail("Invalid print-preparation report"); return
            }
            guard offenders.isEmpty else { self.fail("Page-boundary offenders: \(offenders)"); return }
            guard highlightCount == 3,
                  highlightBackgrounds.allSatisfy({ $0 == "transparent" || $0 == "rgba(0, 0, 0, 0)" }) else {
                self.fail("Revision highlights retained screen background in print export: \(highlightBackgrounds)"); return
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
        for required in ["Introduction", "Supplementary Table S1", "Supplementary Table S2", "Supplementary Table S3", "Supplementary Figure S1", "Supplementary Figure S2"] {
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
let configuredOutputPath = ProcessInfo.processInfo.environment["MED_DICT_PRINT_OUTPUT_PATH"]
let outputURL = URL(fileURLWithPath: configuredOutputPath ?? "/private/tmp/med_dict_workshop_print_\(ProcessInfo.processInfo.processIdentifier).pdf")
try? FileManager.default.removeItem(at: outputURL)
let application = NSApplication.shared
application.setActivationPolicy(.prohibited)
let verifier = PrintVerifier(outputURL: outputURL)
verifier.webView.loadFileURL(inputURL, allowingReadAccessTo: projectRoot)
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
