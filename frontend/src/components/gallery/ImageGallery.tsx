import { useState, useEffect, useRef, useCallback } from "react";

/**
 * Gallery image item
 */
export interface GalleryImage {
  id: string;
  src: string;
  alt?: string;
  width?: number;
  height?: number;
}

/**
 * Props for ImageGallery component
 */
interface ImageGalleryProps {
  images: GalleryImage[];
  columns?: number;
  gap?: number;
  onSelect?: (selectedIds: string[]) => void;
}

/**
 * Responsive masonry grid image gallery with lazy loading
 *
 * Uses IntersectionObserver for efficient lazy loading
 */
export function ImageGallery({
  images,
  columns = 3,
  gap = 16,
  onSelect,
}: ImageGalleryProps): JSX.Element {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Setup IntersectionObserver for lazy loading
  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const imageId = entry.target.getAttribute("data-image-id");
            if (imageId && !loadedImages.has(imageId)) {
              setLoadedImages((prev) => new Set([...prev, imageId]));
            }
          }
        });
      },
      {
        rootMargin: "100px",
        threshold: 0.01,
      },
    );

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [loadedImages]);

  // Observe image containers
  const imageRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (node && observerRef.current) {
        observerRef.current.observe(node);
      }
    },
    [],
  );

  const handleImageClick = (id: string, event: React.MouseEvent): void => {
    if (event.ctrlKey || event.metaKey) {
      // Multi-select with Ctrl/Cmd
      const newSelected = new Set(selectedIds);
      if (newSelected.has(id)) {
        newSelected.delete(id);
      } else {
        newSelected.add(id);
      }
      setSelectedIds(newSelected);
      onSelect?.(Array.from(newSelected));
    } else {
      // Single select
      const newSelected = new Set([id]);
      setSelectedIds(newSelected);
      onSelect?.(Array.from(newSelected));
    }
  };

  const handleSelectAll = (): void => {
    const allIds = new Set(images.map((img) => img.id));
    setSelectedIds(allIds);
    onSelect?.(Array.from(allIds));
  };

  const handleDeselectAll = (): void => {
    setSelectedIds(new Set());
    onSelect?.([]);
  };

  // Calculate masonry columns
  const columnArrays: GalleryImage[][] = Array.from(
    { length: columns },
    () => [],
  );
  images.forEach((image, index) => {
    columnArrays[index % columns].push(image);
  });

  return (
    <div>
      {/* Selection Controls */}
      {onSelect && (
        <div
          style={{
            marginBottom: "1rem",
            display: "flex",
            gap: "0.5rem",
            alignItems: "center",
          }}
        >
          <span style={{ fontSize: "0.875rem", color: "#666" }}>
            {selectedIds.size} selected
          </span>
          <button
            onClick={handleSelectAll}
            style={{
              padding: "0.25rem 0.5rem",
              fontSize: "0.75rem",
              backgroundColor: "#f0f0f0",
              border: "1px solid #ddd",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Select All
          </button>
          {selectedIds.size > 0 && (
            <button
              onClick={handleDeselectAll}
              style={{
                padding: "0.25rem 0.5rem",
                fontSize: "0.75rem",
                backgroundColor: "#f0f0f0",
                border: "1px solid #ddd",
                borderRadius: "4px",
                cursor: "pointer",
              }}
            >
              Deselect All
            </button>
          )}
        </div>
      )}

      {/* Masonry Grid */}
      <div
        style={{
          display: "flex",
          gap: `${gap}px`,
        }}
      >
        {columnArrays.map((columnImages, columnIndex) => (
          <div
            key={columnIndex}
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              gap: `${gap}px`,
            }}
          >
            {columnImages.map((image) => {
              const isSelected = selectedIds.has(image.id);
              const isLoaded = loadedImages.has(image.id);

              return (
                <div
                  key={image.id}
                  ref={imageRef}
                  data-image-id={image.id}
                  onClick={(e) => handleImageClick(image.id, e)}
                  style={{
                    position: "relative",
                    cursor: "pointer",
                    border: isSelected ? "3px solid #007bff" : "1px solid #ddd",
                    borderRadius: "8px",
                    overflow: "hidden",
                    backgroundColor: "#f5f5f5",
                    transition: "transform 0.2s, border-color 0.2s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "scale(1.02)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "scale(1)";
                  }}
                >
                  {isLoaded ? (
                    <img
                      src={image.src}
                      alt={image.alt || image.id}
                      style={{
                        width: "100%",
                        height: "auto",
                        display: "block",
                      }}
                      loading="lazy"
                    />
                  ) : (
                    <div
                      style={{
                        width: "100%",
                        paddingBottom: "100%",
                        backgroundColor: "#e0e0e0",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <span style={{ color: "#999" }}>Loading...</span>
                    </div>
                  )}

                  {/* Selection Indicator */}
                  {isSelected && (
                    <div
                      style={{
                        position: "absolute",
                        top: "8px",
                        right: "8px",
                        width: "24px",
                        height: "24px",
                        backgroundColor: "#007bff",
                        borderRadius: "50%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "white",
                        fontSize: "14px",
                        fontWeight: "bold",
                      }}
                    >
                      ✓
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {images.length === 0 && (
        <div
          style={{
            padding: "3rem",
            textAlign: "center",
            color: "#666",
            border: "2px dashed #ddd",
            borderRadius: "8px",
          }}
        >
          <p>No images to display</p>
        </div>
      )}
    </div>
  );
}
